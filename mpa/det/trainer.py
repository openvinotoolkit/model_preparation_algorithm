# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numbers
import os
import os.path as osp
import time
import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.utils import collect_env

from mpa.registry import STAGES
from mpa.modules.utils.task_adapt import extract_anchor_ratio
from mpa.utils.logger import get_logger
from .stage import DetectionStage, build_detector

logger = get_logger()


@STAGES.register_module()
class DetectionTrainer(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for detection

        - Configuration
        - Environment setup
        - Run training via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        model_builder = kwargs.get("model_builder", build_detector)
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)
        logger.info('train!')

        # # Work directory
        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        distributed = False
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                distributed = True

        logger.info(f'cfg.gpu_ids = {cfg.gpu_ids}, distributed = {distributed}')
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        if hasattr(cfg, 'hparams'):
            if cfg.hparams.get('adaptive_anchor', False):
                num_ratios = cfg.hparams.get('num_anchor_ratios', 5)
                proposal_ratio = extract_anchor_ratio(datasets[0], num_ratios)
                self.configure_anchor(cfg, proposal_ratio)

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
            target_classes = cfg.task_adapt.get('final', [])
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed
        meta['exp_name'] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes)
            if 'proposal_ratio' in locals():
                cfg.checkpoint_config.meta.update({'anchor_ratio': proposal_ratio})

        if distributed:
            if cfg.dist_params.get('linear_scale_lr', False):
                new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
                logger.info(f'enabled linear scaling rule to the learning rate. \
                    changed LR from {cfg.optimizer.lr} to {new_lr}')
                cfg.optimizer.lr = new_lr

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.py'))
        # logger.info(f'Config:\n{cfg.pretty_text}')

        DetectionTrainer.configure_fp16_optimizer(cfg, distributed)

        if distributed:
            os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = cfg.dist_params.get('master_port', '29500')
            mp.spawn(DetectionTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                     args=(target_classes, datasets, cfg, model_builder, distributed, True, timestamp, meta))
        else:
            DetectionTrainer.train_worker(
                None,
                target_classes,
                datasets,
                cfg,
                model_builder,
                distributed,
                True,
                timestamp,
                meta)

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, 'latest.pth')
        best_ckpt_path = glob.glob(osp.join(cfg.work_dir, 'best_*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        # NNCF model
        compression_state_path = osp.join(cfg.work_dir, "compression_state.pth")
        if not os.path.exists(compression_state_path):
            compression_state_path = None
        before_ckpt_path = osp.join(cfg.work_dir, "before_training.pth")
        if not os.path.exists(before_ckpt_path):
            before_ckpt_path = None
        return dict(
            final_ckpt=output_ckpt_path,
            compression_state_path=compression_state_path,
            before_ckpt_path=before_ckpt_path,
        )

    @staticmethod
    def train_worker(gpu, target_classes, datasets, cfg, model_builder=None, distributed=False,
                     validate=False, timestamp=None, meta=None):
        if distributed:
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend=cfg.dist_params.get('backend', 'nccl'),
                                    world_size=len(cfg.gpu_ids), rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        # build the model and load checkpoint
        if model_builder is None:
            model_builder = build_detector
        model = model_builder(cfg)
        model.CLASSES = target_classes

        DetectionTrainer.configure_unlabeled_dataloader(
            cfg,
            build_dataset,
            build_dataloader,
            distributed
        )

        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
