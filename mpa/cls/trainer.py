# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import numbers
import os.path as osp
import time
import copy
import warnings
import torch
import numpy as np
import random

import torch.multiprocessing as mp
import torch.distributed as dist

import mmcv

from mmcls import __version__
from mmcls.apis import train_model
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.utils import collect_env

from mpa.registry import STAGES
from mpa.cls.stage import ClsStage, build_classifier
from mpa.utils.logger import get_logger


logger = get_logger()


@STAGES.register_module()
class ClsTrainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        model_builder = kwargs.get("model_builder", build_classifier)
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # Environment
        distributed = False
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                distributed = True

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)

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

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed

        if isinstance(datasets[0], list):
            repr_ds = datasets[0][0]
        else:
            repr_ds = datasets[0]

        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmcls_version=__version__)
            if hasattr(repr_ds, 'tasks'):
                cfg.checkpoint_config.meta['tasks'] = repr_ds.tasks
            else:
                cfg.checkpoint_config.meta['CLASSES'] = repr_ds.CLASSES
            if 'task_adapt' in cfg:
                if hasattr(self, 'model_tasks'):  # for incremnetal learning
                    cfg.checkpoint_config.meta.update({'tasks': self.model_tasks})
                    # instead of update(self.old_tasks), update using "self.model_tasks"
                else:
                    cfg.checkpoint_config.meta.update({'CLASSES': self.model_classes})

        if distributed:
            if cfg.dist_params.get('linear_scale_lr', False):
                new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
                logger.info(f'enabled linear scaling rule to the learning rate. \
                    changed LR from {cfg.optimizer.lr} to {new_lr}')
                cfg.optimizer.lr = new_lr

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.yaml')) # FIXME bug to save
        # logger.info(f'Config:\n{cfg.pretty_text}')

        if distributed:
            os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = cfg.dist_params.get('master_port', '29500')

            mp.spawn(ClsTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                     args=(datasets, cfg, model_builder, distributed, True, timestamp, meta))
        else:
            ClsTrainer.train_worker(None, datasets, cfg, model_builder,
                                    distributed,
                                    True,
                                    timestamp,
                                    meta)

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, 'best_model.pth'
                                    if osp.exists(osp.join(cfg.work_dir, 'best_model.pth'))
                                    else 'latest.pth')
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
    def train_worker(gpu, dataset, cfg, model_builder, distributed, validate, timestamp, meta):
        logger.info(f'called train_worker() gpu={gpu}, distributed={distributed}, validate={validate}')
        if distributed:
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend=cfg.dist_params.get('backend', 'nccl'),
                                    world_size=len(cfg.gpu_ids), rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        # build the model and load checkpoint
        if model_builder is None:
            model_builder = build_classifier
        model = model_builder(cfg)

        ClsTrainer.configure_custom_fp16_optimizer(cfg, distributed)
        ClsTrainer.configure_unlabeled_dataloader(
            cfg,
            build_dataset,
            build_dataloader,
            distributed
        )

        # register custom eval hooks
        if validate:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_loader_cfg = {
                "samples_per_gpu": cfg.data.samples_per_gpu,
                "workers_per_gpu": cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                "num_gpus": len(cfg.gpu_ids),
                "dist": distributed,
                "round_up": True,
                "seed": cfg.seed,
                "shuffle": False,     # Not shuffle by default
                "sampler_cfg": None,  # Not use sampler by default
                **cfg.data.get('val_dataloader', {}),
            }
            val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            cfg.custom_hooks.append(
                dict(
                    type="DistCustomEvalHook" if distributed else "CustomEvalHook",
                    dataloader=val_dataloader,
                    priority='ABOVE_NORMAL',
                    **eval_cfg,
                )
            )

        train_model(
            model=model,
            dataset=dataset,
            cfg=cfg,
            distributed=distributed,
            validate=False,
            timestamp=timestamp,
            meta=meta,
        )
