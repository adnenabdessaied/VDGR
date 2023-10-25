import os
import os.path as osp
import numpy as np
import json
import argparse
import pyhocon
import glog as log
import torch
from tqdm import tqdm

from utils.data_utils import load_pickle_lines
from utils.visdial_metrics import scores_to_ranks


parser = argparse.ArgumentParser(description='Ensemble for VisDial')
parser.add_argument('--exp', type=str, default='test',
                    help='experiment name from .conf')
parser.add_argument('--mode', type=str, default='predict', choices=['eval', 'predict'],
                    help='eval or predict')
parser.add_argument('--ssh', action='store_true',
                    help='whether or not we are executing command via ssh. '
                         'If set to True, we will not log.info anything to screen and only redirect them to log file')


if __name__ == '__main__':
    args = parser.parse_args()

    # initialization
    config = pyhocon.ConfigFactory.parse_file(f"config/ensemble.conf")[args.exp]
    config["log_dir"] = os.path.join(config["log_dir"], args.exp)
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    # set logs
    log_file = os.path.join(config["log_dir"], f'{args.mode}.log')
    set_log_file(log_file, file_only=args.ssh)

    # print environment info
    log.info(f"Running experiment: {args.exp}")
    log.info(f"Results saved to {config['log_dir']}")
    log.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    if isinstance(config['processed'], list):
        assert len(config['models']) == len(config['processed'])
        processed = {model:pcd for model, pcd in zip(config['models'], config['processed'])}
    else:
        processed = {model: config['processed'] for model in config['models']}

    if config['split'] == 'test' and np.any(config['processed']):
        test_data = json.load(open(config['visdial_test_data']))['data']['dialogs']
        imid2rndid = {t['image_id']: len(t['dialog']) for t in test_data}
        del test_data

    # load predictions files
    visdial_outputs = dict()
    if args.mode == 'eval':
        metrics = {}
    for model in config['models']:
        pred_filename = osp.join(config['pred_dir'], model, 'visdial_prediction.pkl')
        pred_dict = {p['image_id']: p for p in load_pickle_lines(pred_filename)}
        log.info(f'Loading {len(pred_dict)} predictions from {pred_filename}')
        visdial_outputs[model] = pred_dict
        if args.mode == 'eval':
            assert len(visdial_outputs[model]) >= num_dialogs
            metric = json.load(open(osp.join(config['pred_dir'], model, "metrics_epoch_best.json")))
            metrics[model] = metric['val']

    image_ids = visdial_outputs[model].keys()
    predictions = []

    # for each dialog
    for image_id in tqdm(image_ids):
        scores = []
        round_id = None
        
        for model in config['models']:
            pred = visdial_outputs[model][image_id]
            
            if config['split'] == 'test' and processed[model]:
                # if predict on processed data, the first few rounds are deleted from some dialogs
                # so the original round ids can only be found in the original test data
                round_id_in_pred = imid2rndid[image_id]
            else:
                round_id_in_pred = pred['gt_relevance_round_id']

            if not isinstance(round_id_in_pred, int):
                round_id_in_pred = int(round_id_in_pred)
            if round_id is None:
                round_id = round_id_in_pred
            else:
                # make sure all models have the same round_id
                assert round_id == round_id_in_pred
            scores.append(torch.from_numpy(pred['nsp_probs']).unsqueeze(0))

        # ensemble scores
        scores = torch.cat(scores, 0) # [n_model, num_rounds, num_options]
        scores = torch.sum(scores, dim=0, keepdim=True) # [1, num_rounds, num_options]

      
        if scores.size(0) > 1:
            scores = scores[round_id - 1].unsqueeze(0)
        ranks = scores_to_ranks(scores) # [eval_batch_size, num_rounds, num_options]
        ranks = ranks.squeeze(1)
        prediction = {
            "image_id": image_id,
            "round_id": round_id,
            "ranks": ranks[0].tolist()
        }
        predictions.append(prediction)

    filename = osp.join(config['log_dir'], f'{config["split"]}_ensemble_preds.json')
    with open(filename, 'w') as f:
        json.dump(predictions, f)
    log.info(f'{len(predictions)} predictions saved to {filename}')
