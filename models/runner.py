import os
import os.path as osp
import json
from collections import deque
import time
import re
import shutil
import glob
import pickle
import gc
import numpy as np
import glog as log
try:
    from apex import amp
except ModuleNotFoundError:
    print('apex not found')

import torch
import torch.utils.data as tud
import torch.nn.functional as F
import torch.distributed as dist

from utils.data_utils import load_pickle_lines
from utils.visdial_metrics import SparseGTMetrics, NDCG, scores_to_ranks
import wandb


class Runner:
    def __init__(self, config):
        self.config = config
        if 'rank' in config:
            self.gpu_rank = config['rank']
        else:
            self.gpu_rank = 0

        self.epoch_idx = 0
        self.max_metric = 0.
        self.max_metric_epoch_idx = 0
        self.na_str = 'N/A'

        if self.config["max_ckpt_to_keep"] > 0:
            self.checkpoint_queue = deque(
                [], maxlen=config["max_ckpt_to_keep"])
            self.metrics_queue = deque([], maxlen=config["max_ckpt_to_keep"])

        self.setup_wandb()

    def setup_wandb(self):
        if self.gpu_rank == 0:
            print("[INFO] Set wandb logging on rank {}".format(0))
            run = wandb.init(
                project=self.config['wandb_project'], config=self.config, mode=self.config['wandb_mode'])
        else:
            run = None
        self.run = run

    def forward(self, batch, eval_visdial=False):
        return NotImplementedError

    def train(self, dataset, dataset_eval=None):
        # wandb.login()
        if os.path.exists(self.config['log_dir']) or self.config['loads_ckpt'] or self.config['loads_best_ckpt']:
            self.load_ckpt()

        if self.config['use_trainval']:
            dataset.split = 'trainval'
        else:
            dataset.split = 'train'
        batch_size = self.config['batch_size']
        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            sampler_tr = tud.distributed.DistributedSampler(
                dataset,
                num_replicas=self.config['num_gpus'],
                rank=self.gpu_rank
            )
        else:
            sampler_tr = None

        data_loader_tr = tud.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=self.config['training'] and not self.config['parallel'],
            collate_fn=dataset.collate_fn,
            num_workers=self.config['num_workers'],
            sampler=sampler_tr
        )

        
        start_epoch_idx = self.epoch_idx
        num_iter_epoch = self.config['num_iter_per_epoch']
        if self.config['display']:
            log.info(f'{num_iter_epoch} iter per epoch.')

        # eval before training
        eval_dense_at_first = self.config['train_on_dense'] and self.config['skip_mrr_eval'] and start_epoch_idx == 0
        # eval before training under 2 circumstances:
        # for dense finetuning, eval ndcg before the first epoch
        # for mrr training, continue training and the last epoch is not evaluated
        
        if (eval_dense_at_first or (self.config['eval_at_start'] and len(self.metrics_queue) == 0 and start_epoch_idx > 0)):
            if eval_dense_at_first:
                iter_now = 0
            else:
                iter_now = max(num_iter_epoch * start_epoch_idx, 0)
            
            if dataset_eval is None:
                dataset.split = 'val'
                dataset_to_eval = dataset
            else:
                dataset_to_eval = dataset_eval

            metrics_results = {}
            metrics_to_maximize, metrics_results['val'] = self.evaluate(
                dataset_to_eval, iter_now)
            if eval_dense_at_first:
                self.max_metric = metrics_to_maximize
                self.max_metric_epoch_idx = -1
            else:
                if self.config['display']:
                    self.save_eval_results(
                        'val', start_epoch_idx - 1, metrics_results)
                    if metrics_to_maximize > self.max_metric:
                        self.max_metric = metrics_to_maximize
                        self.max_metric_epoch_idx = start_epoch_idx - 1
                        self.copy_best_results('val', start_epoch_idx - 1)
                        self.copy_best_predictions('val')
            if dataset_eval is None:
                if self.config['use_trainval']:
                    dataset.split = 'trainval'
                else:
                    dataset.split = 'train'

        num_epochs = self.config['num_epochs']

        for epoch_idx in range(start_epoch_idx, num_epochs):
            if self.config['parallel'] and self.config['dp_type'] != 'dp':
                sampler_tr.set_epoch(epoch_idx)

            self.epoch_idx = epoch_idx

            if self.config['display']:
                log.info(f'starting epoch {epoch_idx}')
                log.info('training')

            self.model.train()

            num_batch = 0
            next_logging_pct = .1
            next_evaluating_pct = self.config["next_evaluating_pct"] + .1
            start_time = time.time()
            self.optimizer.zero_grad()

            for batch in data_loader_tr:
                if self.config['eval_before_training']:
                    log.info('Skipping stright to evaluation...')
                    break
                num_batch += 1
                pct = num_batch / num_iter_epoch * 100
                iter_now = num_iter_epoch * epoch_idx + num_batch

                output = self.forward(batch)
                losses = output['losses']

                # optimizer step
                losses['tot_loss'] /= self.config['batch_multiply']
                # debug
                if self.config['debugging']:
                    log.info('try backward')
                if self.config['dp_type'] == 'apex':
                    with amp.scale_loss(losses['tot_loss'], self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    losses['tot_loss'].backward()
                if self.config['debugging']:
                    log.info('backward done')

                if iter_now % self.config['batch_multiply'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                # display and eval
                if pct >= next_logging_pct:
                    if self.config['display']:
                        loss_to_print = ''
                        for key in losses:
                            if losses[key] is not None and isinstance(losses[key], torch.Tensor):
                                loss_to_print += f'[{key}: {losses[key].item():.4f}]'
                        print(
                            f'[{int(pct)}%][Epoch: {epoch_idx + 1}/{num_epochs}][Iter : {num_batch}/{len(data_loader_tr)}] [time: {time.time() - start_time:.2f}] {loss_to_print}'
                        )


                    next_logging_pct += self.config["next_logging_pct"]

                    if self.config['debugging']:
                        break

                if pct >= next_evaluating_pct: 
                    next_evaluating_pct += self.config["next_evaluating_pct"]

                if self.run:
                    if self.config['train_on_dense']:
                        self.run.log(
                            {
                                "Train/dense_loss": losses['dense_loss'],
                                "Train/total_loss": losses['tot_loss'],
                            },
                            step=iter_now
                        )

                    else:
                        self.run.log(
                            {
                                "Train/lm_loss": losses['lm_loss'],
                                "Train/img_loss": losses['img_loss'],
                                "Train/nsp_loss": losses['nsp_loss'],
                                "Train/total_loss": losses['tot_loss'],
                            },
                            step=iter_now
                        )

                    lr_gnn, lr_bert = self.scheduler.get_lr()[0], self.scheduler.get_lr()[1]
                    self.run.log(
                        {
                            "Train/lr_gnn": lr_gnn,
                            "Train/lr_bert": lr_bert,
                        },
                        step=iter_now
                    )
                    del losses
                # debug
            torch.cuda.empty_cache()

            if self.config['display']:
                log.info(
                    f'100%,\ttime:\t{time.time() - start_time:.2f}'
                )
                ckpt_path = self.save_ckpt()

            if not self.config['skip_visdial_eval'] and self.epoch_idx % self.config['eval_visdial_every'] == 0:

                iter_now = num_iter_epoch * (epoch_idx + 1)

                if dataset_eval is None:
                    dataset.split = 'val'
                    dataset_to_eval = dataset
                else:
                    dataset_to_eval = dataset_eval
                metrics_results = {}
                metrics_to_maximize, metrics_results['val'] = self.evaluate(
                    dataset_to_eval, iter_now)
                if dataset_eval is None:
                    if self.config['use_trainval']:
                        dataset.split = 'trainval'
                    else:
                        dataset.split = 'train'
                if self.config['display']:
                    self.save_eval_results('val', epoch_idx, metrics_results)

                if self.config['display']:
                   
                    if metrics_to_maximize > self.max_metric:
                        self.max_metric = metrics_to_maximize
                        self.max_metric_epoch_idx = epoch_idx
                        self.copy_best_results('val', epoch_idx)
                        self.copy_best_predictions('val')

                    elif not self.config['parallel'] and epoch_idx - self.max_metric_epoch_idx > self.config["early_stop_epoch"]:
                        log.info('Early stop.')
                        break

                    if self.run:
                        self.run.log(
                            {"Val/metric_best": self.max_metric}, step=iter_now)

            if self.config['parallel']:
                if self.config['dp_type'] == 'dp':
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    dist.barrier()
                    log.info('Rank {} passed barrier...'.format(self.gpu_rank))

            if self.config['stop_epochs'] >= 0 and epoch_idx + 1 >= self.config['stop_epochs']:
                if self.config['display']:
                    log.info('Stop for reaching stop_epochs.')
                break

    def evaluate(self, dataset, training_iter=None, eval_visdial=True):
        # create files to save output
        if self.config['predicting']:
            visdial_file_name = None
            if self.config['save_score']:
                visdial_file_name = osp.join(
                    self.config['log_dir'], f'visdial_prediction.pkl')
                if osp.exists(visdial_file_name):
                    dialogs_predicted = load_pickle_lines(
                        visdial_file_name)
                    dialogs_predicted = [d['image_id']
                                            for d in dialogs_predicted]
                else:
                    dialogs_predicted = []
                f_visdial = open(visdial_file_name, 'ab')

            else:
                visdial_file_name = osp.join(
                    self.config['log_dir'], f'visdial_prediction.jsonlines')
                if self.config['parallel'] and self.config['dp_type'] != 'dp':
                    visdial_file_name = visdial_file_name.replace(
                        '.jsonlines', f'_{self.config["rank"]}of{self.config["num_gpus"]}.jsonlines')
                if osp.exists(visdial_file_name):
                    dialogs_predicted_visdial = [json.loads(
                        line)['image_id'] for line in open(visdial_file_name)]
                    f_visdial = open(visdial_file_name, 'a')
                else:
                    dialogs_predicted_visdial = []
                    f_visdial = open(visdial_file_name, 'w')

                dialogs_predicted = dialogs_predicted_visdial

            if len(dialogs_predicted) > 0:
                log.info(f'Found {len(dialogs_predicted)} predicted results.')

            if self.config['display']:
                if visdial_file_name is not None:
                    log.info(
                        f'VisDial predictions saved to {visdial_file_name}')

        elif self.config['display']:
            if self.config['continue_evaluation']:
                predicted_files = os.listdir(
                    osp.join(self.config['visdial_output_dir'], dataset.split))
                dialogs_predicted = [
                    int(re.match(r'(\d+).npz', p).group(1)) for p in predicted_files]
            else:
                if osp.exists(osp.join(self.config['visdial_output_dir'], dataset.split)):
                    shutil.rmtree(
                        osp.join(self.config['visdial_output_dir'], dataset.split))
                os.makedirs(
                    osp.join(self.config['visdial_output_dir'], dataset.split))

                dialogs_predicted = []
            log.info(f'Found {len(dialogs_predicted)} predicted results.')
        
        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            sampler_val = tud.distributed.DistributedSampler(
                            dataset,
                            num_replicas=self.config['num_gpus'],
                            rank=self.gpu_rank
                        )

            sampler_val.set_epoch(self.epoch_idx)
        else:
            sampler_val = None

        data_loader_val = tud.DataLoader(
            dataset=dataset,
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=self.config['num_workers'],
            sampler=sampler_val
        )
        self.model.eval()

        with torch.no_grad():
            if self.config['display']:
                log.info(f'Evaluating {len(dataset)} samples')

            next_logging_pct = self.config["next_logging_pct"] + .1
            if self.config['parallel'] and self.config['dp_type'] == 'dp':
                num_batch_tot = int(
                    np.ceil(len(dataset) / self.config['eval_batch_size']))
            else:
                num_batch_tot = int(np.ceil(
                    len(dataset) / (self.config['eval_batch_size'] * self.config['num_gpus'])))
            num_batch = 0
            if dataset.split == 'val':
                num_options = self.config["num_options"]
                if self.config['skip_mrr_eval']:
                    num_rounds = 1
                else:
                    num_rounds = 10
            elif dataset.split == 'test':
                num_options = 100
                num_rounds = 1
            if self.gpu_rank == 0:
                start_time = time.time()

            for batch in data_loader_val:
                num_batch += 1
                # skip dialogs that have been predicted
                if self.config['predicting']:
                    image_ids = batch['image_id'].tolist()
                    skip_batch = True
                    for image_id in image_ids:
                        if image_id not in dialogs_predicted:
                            skip_batch = False
                    if skip_batch:
                        continue
                output = self.forward(
                    batch, eval_visdial=eval_visdial)

                # visdial evaluation
                if eval_visdial:
                    img_ids = batch['image_id'].tolist()
                    batch_size = len(img_ids)
                    if not self.config['skip_ndcg_eval']:
                        gt_relevance_round_id = batch['round_id'].tolist()

                    # [batch_size * num_rounds * num_options, 2]
                    nsp_scores = output['nsp_scores']
                    nsp_probs = F.softmax(nsp_scores, dim=1)
                    assert nsp_probs.shape[-1] == 2
                    # num_dim=2, 0 for postive, 1 for negative
                    nsp_probs = nsp_probs[:, 0]
                    nsp_probs = nsp_probs.view(
                        batch_size, num_rounds, num_options)

                    # could be predicting or evaluating
                    if dataset.split == 'val':
                        if self.config['skip_ndcg_eval']:
                            gt_option_inds = batch['gt_option_inds']

                            for b in range(batch_size):
                                filename = osp.join(
                                    self.config['visdial_output_dir'], dataset.split, f'{img_ids[b]}.npz')
                                if not osp.exists(filename):
                                    np.savez(
                                        filename,
                                        nsp_probs=nsp_probs[b].cpu().numpy(),
                                        gt_option_inds=gt_option_inds[b].cpu().numpy()
                                    )
                        else:
                            # [batch_size, num_rounds]
                            gt_option_inds = batch['gt_option_inds']
                            # [batch_size, num_options]
                            gt_relevance = batch['gt_relevance']

                            for b in range(batch_size):
                                filename = osp.join(
                                    self.config['visdial_output_dir'], dataset.split, f'{img_ids[b]}.npz')
                                if not osp.exists(filename):
                                    np.savez(filename,
                                            nsp_probs=nsp_probs[b].cpu().numpy(),
                                            gt_option_inds=gt_option_inds[b].cpu(
                                            ).numpy(),
                                            gt_relevance=gt_relevance[b].cpu(
                                            ).numpy(),
                                            gt_relevance_round_id=gt_relevance_round_id[b])

                    # must be predicting
                    if dataset.split == 'test':
                        if self.config['save_score']:
                            for b in range(batch_size):
                                prediction = {
                                    "image_id": img_ids[b],
                                    "nsp_probs": nsp_probs[b].cpu().numpy(),
                                    "gt_relevance_round_id": gt_relevance_round_id[b]
                                }
                                pickle.dump(prediction, f_visdial)
                        else:
                            # [eval_batch_size, num_rounds, num_options]
                            ranks = scores_to_ranks(nsp_probs)
                            ranks = ranks.squeeze(1)
                            for b in range(batch_size):
                                prediction = {
                                    "image_id": img_ids[b],
                                    "round_id": gt_relevance_round_id[b],
                                    "ranks": ranks[b].tolist()
                                }
                                f_visdial.write(json.dumps(prediction) + '\n')
            
                # debug
                if self.config['debugging']:
                    break

                pct = num_batch / num_batch_tot * 100
                if pct >= next_logging_pct:
                    if self.config['display'] and self.gpu_rank == 0:
                        log.info(
                            f'{int(pct)}%,\ttime:\t{time.time() - start_time:.2f}'
                        )
                    next_logging_pct += self.config["next_logging_pct"]
                    # debug
                    if self.config['debugging']:
                        break

        if self.config['display'] and self.gpu_rank == 0:
            pct = num_batch / num_batch_tot * 100
            log.info(
                f'{int(pct)}%,\ttime:\t{time.time() - start_time:.2f}'
            )

        if not self.config['validating']:
            self.model.train()

        if self.config['parallel'] and self.config['dp_type'] != 'dp':
            dist.barrier()

        print(f'{self.gpu_rank} passed barrier')

        if self.config['predicting']:
            f_visdial.close()
            if not self.config['save_score']:
                all_visdial_predictions = [json.loads(
                    line) for line in open(visdial_file_name)]
                if self.config['predict_split'] == 'test' and len(all_visdial_predictions) == self.config['num_test_dialogs']:
                    visdial_file_name = visdial_file_name.replace(
                        'jsonlines', 'json')
                    with open(visdial_file_name, 'w') as f_visdial:
                        json.dump(all_visdial_predictions, f_visdial)
                    log.info(
                        f'Prediction for submisson save to {visdial_file_name}.')
            return None, None

        if self.config['display']:
            if dataset.split == 'val' and eval_visdial:
                if not self.config['skip_mrr_eval']:
                    sparse_metrics = SparseGTMetrics()
                if not self.config['skip_ndcg_eval']:
                    ndcg = NDCG()

            if dataset.split == 'val' and eval_visdial:
                visdial_output_filenames = glob.glob(
                    osp.join(self.config['visdial_output_dir'], dataset.split, '*.npz'))
                log.info(
                    f'Calculating visdial metrics for {len(visdial_output_filenames)} dialogs')
                for visdial_output_filename in visdial_output_filenames:
                    output = np.load(visdial_output_filename)
                    nsp_probs = torch.from_numpy(
                        output['nsp_probs']).unsqueeze(0)
                    if not self.config['skip_ndcg_eval']:
                        gt_relevance = torch.from_numpy(output['gt_relevance']).unsqueeze(0)
                    if not self.config['skip_mrr_eval']:
                        gt_option_inds = torch.from_numpy(
                            output['gt_option_inds']).unsqueeze(0)
                        sparse_metrics.observe(nsp_probs, gt_option_inds)
                        if not self.config['skip_ndcg_eval']:
                            gt_relevance_round_id = output['gt_relevance_round_id']
                            nsp_probs_dense = nsp_probs[0, gt_relevance_round_id - 1, :].unsqueeze(0)
                    else:
                        nsp_probs_dense = nsp_probs.squeeze(0)  # [1, 100]
                    if not self.config['skip_ndcg_eval']:
                        ndcg.observe(nsp_probs_dense, gt_relevance)

            # visdial eval output
            visdial_metrics = {}
            if dataset.split == 'val' and eval_visdial:
                if not self.config['skip_mrr_eval']:
                    visdial_metrics.update(sparse_metrics.retrieve(reset=True))
                if not self.config['skip_ndcg_eval']:
                    visdial_metrics.update(ndcg.retrieve(reset=True))

                if self.config['display']:
                    to_print = ''
                    for metric_name, metric_value in visdial_metrics.items():
                        if 'round' not in metric_name:
                            to_print += f"\n{metric_name}: {metric_value}"
                            if training_iter is not None:
                                if self.run:
                                    self.run.log(
                                        {'Val/' + metric_name: metric_value}, step=training_iter)
                    log.info(to_print)

            if self.config['metrics_to_maximize'] in visdial_metrics:
                metrics_to_maximize = visdial_metrics[self.config['metrics_to_maximize']]
            else:
                metrics_to_maximize = None
            torch.cuda.empty_cache()
            return metrics_to_maximize, visdial_metrics
        else:
            torch.cuda.empty_cache()
            return None, None

    def save_eval_results(self, split, epoch_idx, metrics_results):

        metrics_filename = osp.join(
            self.config['log_dir'], f'metrics_epoch_{epoch_idx}.json')
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_results, f)
        log.info(f'Results of metrics saved to {metrics_filename}')

        if self.config["max_ckpt_to_keep"] > 0:
            if len(self.metrics_queue) == self.metrics_queue.maxlen:
                todel = self.metrics_queue.popleft()
                os.remove(todel)
            self.metrics_queue.append(metrics_filename)

        if epoch_idx == 'best':
            self.copy_best_predictions(split)

    def copy_best_results(self, split, epoch_idx):
        to_print = 'Copy '

        if not self.config['skip_saving_ckpt']:
            ckpt_path = osp.join(
                self.config['log_dir'], f'epoch_{epoch_idx}.ckpt')
            best_ckpt_path = ckpt_path.replace(
                f'{epoch_idx}.ckpt', 'best.ckpt')
            shutil.copyfile(ckpt_path, best_ckpt_path)
            to_print += best_ckpt_path + ' '

        metrics_filename = osp.join(
            self.config['log_dir'], f'metrics_epoch_{epoch_idx}.json')
        best_metric_filename = metrics_filename.replace(
            f'{epoch_idx}.json', 'best.json')
        shutil.copyfile(metrics_filename, best_metric_filename)
        to_print += best_metric_filename + ' '

        log.info(to_print)

    def copy_best_predictions(self, split):
        to_print = 'Copy '

        visdial_output_dir = osp.join(self.config['visdial_output_dir'], split)
        if osp.exists(visdial_output_dir):
            dir_best = visdial_output_dir.replace('output', 'output_best')
            if osp.exists(dir_best):
                shutil.rmtree(dir_best)
            shutil.copytree(visdial_output_dir, dir_best)
            to_print += dir_best + ' '

        log.info(to_print)

    def get_ckpt(self):
        ckpt = {
            'epoch_idx': self.epoch_idx,
            'max_metric': self.max_metric,
            'seed': self.config['random_seed'],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        if self.config['parallel']:
            ckpt['model_state_dict'] = self.model.module.state_dict()
        else:
            ckpt['model_state_dict'] = self.model.state_dict()
        if self.config['dp_type'] == 'apex':
            ckpt['amp'] = amp.state_dict()
        return ckpt

    def set_ckpt(self, ckpt_dict):
        if not self.config['restarts']:
            self.epoch_idx = ckpt_dict.get('epoch_idx', -1) + 1

        if not self.config['resets_max_metric']:
            self.max_metric = ckpt_dict.get('max_metric', -1)

        if self.config['parallel']:
            model = self.model.module
        else:
            model = self.model

        model_state_dict = model.state_dict()
        former_dict = {
            k: v for k, v in ckpt_dict['model_state_dict'].items() if k in model_state_dict}

        if self.config['display']:
            log.info("number of keys transferred: %d" % len(former_dict))
        assert len(former_dict.keys()) > 0

        model_state_dict.update(former_dict)

        model.load_state_dict(model_state_dict)
        if self.config['display']:
            log.info('loaded model')
        del model_state_dict, former_dict

        if not self.config['validating'] and not (self.config['uses_new_optimizer'] or self.config['sets_new_lr']):
            if 'optimizer' in ckpt_dict:
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
                if self.config['display']:
                    log.info('loaded optimizer')
            if 'scheduler' in ckpt_dict:
                self.scheduler.last_epcoh = ckpt_dict['epoch_idx'] * \
                    self.config['num_iter_per_epoch']
                self.scheduler.load_state_dict(ckpt_dict['scheduler'])

        if 'amp' in ckpt_dict and self.config['dp_type'] == 'apex':
            amp.load_state_dict(ckpt_dict['amp'])

        del ckpt_dict

        torch.cuda.empty_cache()

    def save_ckpt(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_{self.epoch_idx}.ckpt'
        log.info(f'saving checkpoint {ckpt_path}')
        ckpt = self.get_ckpt()
        if self.config['skip_saving_ckpt']:
            return ckpt_path
        torch_version = float(torch.__version__[:3])
        if torch_version - 1.4 > 1e-3:
            torch.save(ckpt, f=ckpt_path, _use_new_zipfile_serialization=False)
        else:
            torch.save(ckpt, f=ckpt_path)
        del ckpt

        if not (self.config['parallel'] and self.config['dp_type'] in ['ddp', 'apex']):
            torch.cuda.empty_cache()

        if self.config["max_ckpt_to_keep"] > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                os.remove(todel)
            self.checkpoint_queue.append(ckpt_path)

        return ckpt_path

    def save_ckpt_best(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
        log.info(f'saving checkpoint {ckpt_path}')
        ckpt = self.get_ckpt()
        torch.save(ckpt, f=ckpt_path)
        del ckpt
        return ckpt_path

    def load_ckpt_best(self):
        ckpt_path = f'{osp.dirname(self.config["log_dir"])}/epoch_best.ckpt'
        if not osp.exists(ckpt_path):
            ckpt_paths = [path for path in os.listdir(
                f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
            if len(ckpt_paths) == 0:
                if self.config['display']:
                    log.info(f'No .ckpt found in {self.config["log_dir"]}')
                return

            def sort_func(x): return int(re.search(r"(\d+)", x).groups()[0])
            ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func, reverse=True)[0]}'
        if self.config['display']:
            log.info(f'loading checkpoint {ckpt_path}')
        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        self.set_ckpt(torch.load(ckpt_path, map_location=map_location))

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if self.config['validating'] or self.config['loads_best_ckpt']:
                ckpt_path = f'{self.config["log_dir"]}/epoch_best.ckpt'
            else:
                ckpt_paths = [path for path in os.listdir(
                    f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
                if len(ckpt_paths) == 0:
                    if self.config['display']:
                        log.info(f'No .ckpt found in {self.config["log_dir"]}')
                    return

                def sort_func(x): return int(
                    re.search(r"(\d+)", x).groups()[0])
                ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func, reverse=True)[0]}'

        if self.config['display']:
            log.info(f'loading checkpoint {ckpt_path}')
            epoch_name = osp.split(ckpt_path)[1].split('.')[0]
            if re.search(r"(\d+)", epoch_name):
                self.checkpoint_queue.append(ckpt_path)
                metrics_filename = osp.join(
                    self.config['log_dir'], f'metrics_{epoch_name}.json')
                if osp.exists(metrics_filename):
                    self.metrics_queue.append(metrics_filename)

        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        self.set_ckpt(torch.load(ckpt_path, map_location=map_location))

    def match_model_key(self, pretrained_dict, model_dict):
        matched_dict = dict()
        for key in pretrained_dict:
            if key in model_dict:
                matched_key = key
            elif key.startswith('encoder.') and key[8:] in model_dict:
                matched_key = key[8:]
            elif key.startswith('module.') and key[7:] in model_dict:
                matched_key = key[7:]
            elif 'encoder.' + key in model_dict:
                matched_key = 'encoder.' + key
            elif 'module.' + key in model_dict:
                matched_key = 'module.' + key
            else:
                # not_found.append(key)
                continue
            matched_dict[matched_key] = pretrained_dict[key]

        not_found = ""
        for k in model_dict:
            if k not in matched_dict:
                not_found += k + '\n'

        log.info("Keys from model_dict that were not found in pretrained_dict:")
        log.info(not_found)
        return matched_dict

    def load_pretrained_vilbert(self, start_from=None):
        if start_from is not None:
            self.config["start_path"] = start_from
        if self.config['training'] or self.config['debugging']:
            ckpt_paths = [path for path in os.listdir(
                f'{self.config["log_dir"]}/') if path.endswith('.ckpt') and 'best' not in path]
            if len(ckpt_paths) > 0:
                if self.config['display']:
                    log.info('Continue training')
                return

        if self.config['display']:
            log.info(
                f'Loading pretrained VilBERT from {self.config["start_path"]}')
        map_location = {'cuda:0': f'cuda:{self.gpu_rank}'}
        pretrained_dict = torch.load(
            self.config['start_path'], map_location=map_location)
        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']
        if self.config['parallel']:
            model = self.model.module
        else:
            model = self.model
        model_dict = model.state_dict()

        matched_dict = self.match_model_key(pretrained_dict, model_dict)

        if self.config['display']:
            log.info("number of keys transferred: %d" % len(matched_dict))
        assert len(matched_dict.keys()) > 0
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

        del pretrained_dict, model_dict, matched_dict
        if not self.config['parallel'] or self.config['dp_type'] == 'dp':
            torch.cuda.empty_cache()

        if self.config['display']:
            log.info(f'Pretrained VilBERT loaded')
