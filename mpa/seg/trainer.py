# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import time
import glob
from os import path as osp
import datetime
from multiprocessing import Pipe, Process
import pynvml
import re
import uuid

import mmcv
from mmcv import get_git_hash

from mmseg import __version__
from .train import train_segmentor
from .builder import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env

from mpa.registry import STAGES
from .stage import SegStage

from mpa.utils.logger import get_logger
from torch import nn

logger = get_logger()


@STAGES.register_module()
class SegTrainer(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for segmentation

        - Configuration
        - Environment setup
        - Run training via MMSegmentation -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)

        if cfg.runner.type == 'IterBasedRunner':
            cfg.runner = dict(type=cfg.runner.type, max_iters=cfg.runner.max_iters)

        logger.info('train!')

        # Work directory
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        logger.info(f'cfg.gpu_ids = {cfg.gpu_ids}, distributed = {self.distributed}')
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # Dataset for HPO
        hp_config = kwargs.get('hp_config', None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)

        # Target classes
        if 'task_adapt' in cfg:
            target_classes = cfg.task_adapt.final
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        meta['seed'] = cfg.seed
        meta['exp_name'] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmseg_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes)

        # run GPU utilization getter process
        parent_conn, child_conn = Pipe()
        p = Process(target=self.calculate_average_gpu_util, args=(cfg.work_dir, child_conn,))
        p.start()

        # Model
        model = build_segmentor(cfg.model)
        model.CLASSES = target_classes

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        start_time = datetime.datetime.now()
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=True,
            timestamp=timestamp,
            meta=meta
        )
        with open(osp.join(cfg.work_dir, f"time_{uuid.uuid4().hex}.txt"), "wt") as f:
            f.write(str(datetime.datetime.now() - start_time))

        # kill GPU utilization getter process
        parent_conn.send(True)
        p.join()

        # Save outputs
        output_ckpt_path = os.path.join(cfg.work_dir, 'latest.pth')
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, 'best_mDice_*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, 'best_mIoU_*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(final_ckpt=output_ckpt_path)

    def _modify_cfg_for_distributed(self, model, cfg):
        nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.dist_params.get('linear_scale_lr', False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(f'enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}')
            cfg.optimizer.lr = new_lr

    @staticmethod
    def calculate_average_gpu_util(work_dir: str, pipe: Pipe):
        pynvml.nvmlInit()
        logger.info("GPU utilzer process start")
        filter = re.compile(r"([-+]?\d+) \%.*([-+]?\d+) \%")

        available_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if available_gpu is None:
            gpu_devices = [i for i in range(pynvml.nvmlDeviceGetCount())]
        else:
            gpu_devices = [int(gpuidx) for gpuidx in available_gpu.split(',')]

        per_gpu_util = {gpuidx : 0 for gpuidx in gpu_devices}
        num_total_util = 0
        while True:
            for gpuidx in gpu_devices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpuidx)
                use = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result = filter.search(str(use))
                if result is not None:
                    per_gpu_util[gpuidx] += float(result.group(1))

            if pipe.poll():
                try:
                    report = pipe.recv()
                    if report:
                        break
                except EOFError:
                    continue

            num_total_util += 1

            time.sleep(1)

        pynvml.nvmlShutdown()

        with open(osp.join(work_dir, "gpu_util.txt"), "wt") as f:
            for gpuidx, gpu_util in per_gpu_util.items():
                f.write(f"#{gpuidx} average GPU util : {gpu_util / num_total_util}\n")
            f.write(f"total average GPU util : {sum(per_gpu_util.values()) / (len(per_gpu_util) * num_total_util)}\n")

        logger.info("GPU utilzer process is done")
