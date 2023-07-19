"""
This code is based on https://github.com/yoshitomo-matsubara/torchdistill/blob/main/examples/legacy/image_classification.py
"""

import argparse
import datetime
import json
import os
import time

import matlab.engine
import numpy as np
import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_binary_object_size
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.registry import get_model

from custom.matlab_eval.ber import connect_matlab, compute_ber, compute_multiple_bers

logger = def_logger.getChild(__name__)


def get_argparser():
    parser = argparse.ArgumentParser(description='Beamforming matrix estimation')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--json', help='json string to overwrite config')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-sync_bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('-test_only', action='store_true', help='Only test the models')
    parser.add_argument('-student_only', action='store_true', help='Test the student model only')
    parser.add_argument('-test_wo_model_only', action='store_true', help='Check BER without models')
    parser.add_argument('-check_data_size', action='store_true', help='Check data size')
    parser.add_argument('-use_v_as_input', action='store_true', help='Use V matrix as input data')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def overwrite_config(org_config, sub_config):
    for sub_key, sub_value in sub_config.items():
        if sub_key in org_config:
            if isinstance(sub_value, dict):
                overwrite_config(org_config[sub_key], sub_value)
            else:
                org_config[sub_key] = sub_value
        else:
            org_config[sub_key] = sub_value


def load_model(model_config, device):
    model = get_model(model_config['name'], **model_config['params'])
    ckpt_file_path = model_config['start_ckpt'] if 'start_ckpt' in model_config else model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)


def train_one_epoch(training_box, uses_v_as_input, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, ch_seeds in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        loss = torch.zeros(1).to(device)
        if uses_v_as_input:
            sample_batch = targets
        for sub_sample_batch, sub_targets in zip(sample_batch, targets):
            sub_sample_batch, sub_targets = sub_sample_batch.float().to(device), sub_targets.float().to(device)
            loss += training_box(sub_sample_batch, sub_targets, supp_dict=None)

        training_box.update_params(loss)
        batch_size = sample_batch[0].shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['sample/s'].update(batch_size / (time.time() - start_time))


@torch.no_grad()
def evaluate(model_wo_ddp, data_loader, uses_v_as_input, device, device_ids, distributed, matlab_eval_config,
             eng, uses_sim_data, checks_data_size=False, log_freq=1000, title=None, header='Test:'):
    model = model_wo_ddp.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith('cuda'):
        model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    input_data_size_list = list()
    bottleneck_size_list = list()
    model.eval()
    dataset = data_loader.dataset
    metric_logger = MetricLogger(delimiter='  ')
    for sample_batch, targets, misc_data in metric_logger.log_every(data_loader, log_freq, header):
        real_v_mat_list, imag_v_mat_list = list(), list()
        real_h_mat_list, imag_h_mat_list = list(), list()
        batch_size = None
        mse = torch.zeros(1).to(device)
        if uses_v_as_input:
            sample_batch = targets
        for i, (sub_sample_batch, sub_targets) in enumerate(zip(sample_batch, targets)):
            if batch_size is None:
                batch_size = sub_sample_batch.shape[0]

            sub_sample_batch = sub_sample_batch.float().to(device, non_blocking=True)
            sub_targets = sub_targets.float().to(device, non_blocking=True)
            if checks_data_size:
                input_data_size_list.append(get_binary_object_size(sub_sample_batch) / batch_size)
                encoded_obj = model_wo_ddp.encode(sub_sample_batch)
                bottleneck_size_list.append(get_binary_object_size(encoded_obj) / batch_size)
                output = model_wo_ddp.decode(encoded_obj)
            else:
                output = model(sub_sample_batch)

            mse += functional.mse_loss(output, sub_targets, reduction='mean')
            real_v_mat, imag_v_mat = dataset.reshape_output(output)
            real_v_mat_list.append(real_v_mat)
            imag_v_mat_list.append(imag_v_mat)
            if not uses_sim_data:
                real_h_mat, imag_h_mat = dataset.reshape_output(misc_data[i])
                real_h_mat_list.append(real_h_mat)
                imag_h_mat_list.append(imag_h_mat)

        misc_data = misc_data.numpy() if uses_sim_data else (real_h_mat_list, imag_h_mat_list)
        bers = compute_ber(real_v_mat_list, imag_v_mat_list, misc_data, matlab_eval_config, eng, uses_sim_data)
        metric_logger.meters['mse'].update(mse.item() / len(targets), n=batch_size)
        metric_logger.meters['ber'].update(np.array(bers).mean(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    mse = metric_logger.mse.global_avg
    ber = metric_logger.ber.global_avg
    logger.info(' * MSE {:.4f}\n'.format(mse))
    logger.info(' * BER {:.4f}\n'.format(ber))
    if checks_data_size:
        logger.info(f'Input data size: {np.mean(input_data_size_list)} [KB]')
        logger.info(f'Bottleneck data size: {np.mean(bottleneck_size_list)} [KB]')
    return ber


def train(teacher_model, student_model, dataset_dict, uses_v_as_input, ckpt_file_path,
          device, device_ids, distributed, eng, config, args):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_ber = float('inf')
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_ber, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    matlab_eval_config = eng.load(train_config['matlab_eval_config'])
    uses_sim_data = train_config.get('uses_sim_data', True)
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        train_one_epoch(training_box, uses_v_as_input, device, epoch, log_freq)
        val_ber = evaluate(student_model, training_box.val_data_loader, uses_v_as_input, device, device_ids,
                           distributed, matlab_eval_config, eng, uses_sim_data, log_freq=log_freq, header='Validation:')
        if val_ber < best_val_ber and is_main_process():
            logger.info('Updating ckpt (Best BER: '
                        '{:.4f} -> {:.4f})'.format(best_val_ber, val_ber))
            best_val_ber = val_ber
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_ber, config, args, ckpt_file_path)
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def evaluate_without_model(data_loader,  matlab_eval_config, eng, uses_sim_data,
                           log_freq=1000, title='Target Evaluation', header='Test:'):
    if title is not None:
        logger.info(title)

    dataset = data_loader.dataset
    metric_logger = MetricLogger(delimiter='  ')
    for sample_batch, targets, misc_data in metric_logger.log_every(data_loader, log_freq, header):
        real_v_mat_list, imag_v_mat_list = list(), list()
        real_h_mat_list, imag_h_mat_list = list(), list()
        batch_size = None
        for i, (sub_sample_batch, sub_targets) in enumerate(zip(sample_batch, targets)):
            if batch_size is None:
                batch_size = sub_sample_batch.shape[0]

            real_v_mat, imag_v_mat = dataset.reshape_output(sub_targets)
            real_v_mat_list.append(real_v_mat)
            imag_v_mat_list.append(imag_v_mat)
            real_h_mat, imag_h_mat = dataset.reshape_output(sub_sample_batch)
            real_h_mat_list.append(real_h_mat)
            imag_h_mat_list.append(imag_h_mat)

        misc_data = misc_data if uses_sim_data else (real_h_mat_list, imag_h_mat_list)
        org_bers, angle_bers = \
            compute_multiple_bers(real_v_mat_list, imag_v_mat_list, misc_data, matlab_eval_config, eng, uses_sim_data)
        metric_logger.meters['org_bers'].update(np.array(org_bers).mean(), n=batch_size)
        metric_logger.meters['angle_bers'].update(np.array(angle_bers).mean(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    org_bers = metric_logger.org_bers.global_avg
    angle_bers = metric_logger.angle_bers.global_avg
    logger.info(' * Original BER (uncompressed) {:.4f}'.format(org_bers))
    logger.info(' * Angle-based BER (compressed) {:.4f}\n'.format(angle_bers))
    return org_bers, angle_bers


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    logger.info('matlab.sys.version: {}'.format(matlab.sys.version))
    cudnn.benchmark = True
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    if args.json is not None:
        logger.info('Overwriting config')
        overwrite_config(config, json.loads(args.json))

    device = torch.device(args.device)
    dataset_dict = util.get_all_datasets(config['datasets'])
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model = load_model(teacher_model_config, device) if teacher_model_config is not None else None
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    ckpt_file_path = student_model_config['ckpt']
    student_model = load_model(student_model_config, device)
    uses_v_as_input = args.use_v_as_input
    eng = connect_matlab()
    if not args.test_only and not args.test_wo_model_only:
        train(teacher_model, student_model, dataset_dict, uses_v_as_input, ckpt_file_path,
              device, device_ids, distributed, eng, config, args)
        student_model_without_ddp =\
            student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    matlab_eval_config = eng.load(test_config['matlab_eval_config'])
    uses_sim_data = test_config.get('uses_sim_data', True)
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, uses_v_as_input, device, device_ids, distributed,
                 matlab_eval_config, eng, uses_sim_data, title='[Teacher: {}]'.format(teacher_model_config['name']))
    if not args.test_wo_model_only:
        evaluate(student_model, test_data_loader, uses_v_as_input, device, device_ids, distributed,
                 matlab_eval_config, eng, uses_sim_data, checks_data_size=args.check_data_size,
                 title='[Student: {}]'.format(student_model_config['name']))
    else:
        evaluate_without_model(test_data_loader, matlab_eval_config, eng, uses_sim_data)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
