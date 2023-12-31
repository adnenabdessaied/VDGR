from utils.init_utils import load_runner, load_dataset, set_random_seed, set_training_steps, initialize_from_env, set_log_file, copy_file_to_log
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch
import os
import sys
import argparse
import pyhocon
import glog as log
import socket
import getpass

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp
except ModuleNotFoundError:
    print('apex not found')

parser = argparse.ArgumentParser(description='Main script for VD-GR')
parser.add_argument(
    '--model',
    type=str,
    default='vdgr/P1',
    help='model name to train or test')

parser.add_argument(
    '--mode',
    type=str,
    default='train',
    help='train, eval, predict or debug')

parser.add_argument(
    '--wandb_project',
    type=str,
    default='VD-GR'
)

parser.add_argument(
    '--wandb_mode',
    type=str,
    default='online',
    choices=['online', 'offline', 'disabled', 'run', 'dryrun']
)

parser.add_argument(
    '--tag',
    type=str,
    default='K2',
    help="Tag to differentiate the different runs"
)

parser.add_argument(
    '--eval_dir',
    type=str,
    default='',
    help="Directory of a trained model to evaluate"
)

parser.add_argument('--ssh', action='store_true',
                    help='whether or not we are executing command via ssh. '
                         'If set to True, we will not log.info anything to screen and only redirect them to log file')


def main(gpu, config, args):
    config['training'] = args.mode == 'train'
    config['validating'] = args.mode == 'eval'
    config['debugging'] = args.mode == 'debug'
    config['predicting'] = args.mode == 'predict'
    config['wandb_project'] = args.wandb_project
    config['wandb_mode'] = args.wandb_mode

    if config['parallel'] and config['dp_type'] != 'dp':
        config['rank'] = gpu
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(config['master_port'])
        dist.init_process_group(
            backend='nccl',
            world_size=config['num_gpus'],
            rank=gpu
        )
        config['display'] = gpu == 0
        if config['dp_type'] == 'apex':
            torch.cuda.set_device(gpu)
    else:
        config['display'] = True
    if config['debugging'] or (config['parallel'] and config['dp_type'] != 'dp'):
        config['num_workers'] = 0
    else:
        config['num_workers'] = 0
    # set logs
    log_file = os.path.join(config["log_dir"], f'{args.mode}.log')
    set_log_file(log_file, file_only=args.ssh)

    # print environment info
    if config['display']:
        log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
            socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
        log.info('Command line is: {}'.format(' '.join(sys.argv)))

        if config['parallel'] and config['dp_type'] != 'dp':
            log.info(
                f'World_size: {config["num_gpus"]}, cur rank: {config["rank"]}')
        log.info(f"Running experiment: {args.model}")
        log.info(f"Results saved to {config['log_dir']}")

    # initialization
    if config['display'] and config['training']:
        copy_file_to_log(config['log_dir'])
    set_random_seed(config['random_seed'])

    device = torch.device(f"cuda:{gpu}")
    if config["use_cpu"]:
        device = torch.device("cpu")
    config['device'] = device

    # prepare dataset
    dataset, dataset_eval = load_dataset(config)

    # set training steps
    if not config['validating'] or config['parallel']:
        config = set_training_steps(config, len(dataset))

    if config['display']:
        log.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    # load runner
    runner = load_runner(config)
    # apex
    if config['dp_type'] == 'apex':
        runner.model, runner.optimizer = amp.initialize(runner.model,
                                                        runner.optimizer,
                                                        opt_level="O1")
    # parallel
    if config['parallel']:
        if config['dp_type'] == 'dp':
            runner.model = nn.DataParallel(runner.model)
            runner.model.to(config['device'])
        elif config['dp_type'] == 'apex':
            runner.model = DDP(runner.model)
        elif config['dp_type'] == 'ddp':
            torch.cuda.set_device(gpu)
            runner.model = runner.model.to(gpu)
            runner.model = nn.parallel.DistributedDataParallel(
                runner.model,
                device_ids=[gpu],
                output_device=gpu,
                find_unused_parameters=True)
        else:
            raise ValueError(f'Unrecognized dp_type: {config["dp_type"]}')

    if config['training'] or config['debugging']:
        runner.load_pretrained_vilbert()
        runner.train(dataset, dataset_eval)
    else:
        if config['loads_start_path']:
            runner.load_pretrained_vilbert()
        else:
            runner.load_ckpt_best()

        metrics_results = {}
        if config['predicting']:
            eval_splits = [config['predict_split']]
        else:
            eval_splits = ['val']
            if config['model_type'] == 'conly' and not config['train_each_round']:
                eval_splits.append('test')
        for split in eval_splits:
            if config['display']:
                log.info(f'Results on {split} split of the best epoch')
            if dataset_eval is None:
                dataset_to_eval = dataset
            else:
                dataset_to_eval = dataset_eval
            dataset_to_eval.split = split
            _, metrics_results[split] = runner.evaluate(
                dataset_to_eval, eval_visdial=True)
        if not config['predicting'] and config['display']:
            runner.save_eval_results(split, 'best', metrics_results)

    if config['parallel'] and config['dp_type'] != 'dp':
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()
    # initialization
    model_type, model_name = args.model.split('/')
    config = initialize_from_env(
        model_name, args.mode, args.eval_dir, model_type, tag=args.tag)
    if config['num_gpus'] > 1:
        config['parallel'] = True
        if config['dp_type'] == 'dp':
            main(0, config, args)
        else:
            mp.spawn(main, nprocs=config['num_gpus'], args=(config, args))
    else:
        config['parallel'] = False
        main(0, config, args)
