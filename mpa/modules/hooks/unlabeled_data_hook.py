# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import get_dist_info
from mmcv.runner import HOOKS, Hook

from mmdet.datasets import build_dataset, build_dataloader

from mpa.modules.datasets.composed_dataloader import ComposedDL
from mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class UnlabeledDataHook(Hook):

    def __init__(
        self,
        unlabeled_data_cfg,
        samples_per_gpu,
        workers_per_gpu,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Build unlabeled dataset & loader
        logger.info('In UnlabeledDataHook.before_epoch, creating unlabeled dataset...')
        self.unlabeled_dataset = build_dataset(unlabeled_data_cfg)
        _, world_size = get_dist_info()
        logger.info('In UnlabeledDataHook.before_epoch, creating unlabeled data_loader...')
        self.unlabeled_loader = build_dataloader(
            self.unlabeled_dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=world_size,
            dist=(world_size > 1),
            seed=seed,
            **kwargs
        )
        self.composed_loader = None

    def before_epoch(self, runner):
        if self.composed_loader is None:
            logger.info('In UnlabeledDataHook.before_epoch, creating ComposedDL'
                        f'([labeled({len(runner.data_loader)}, unlabeled({len(self.unlabeled_loader)})])')
            self.composed_loader = ComposedDL([runner.data_loader, self.unlabeled_loader])
        # Per-epoch replacement: train-only loader -> train+unlabeled loader
        # (It's similar to local variable in epoch. Need to update every epoch...)
        runner.data_loader = self.composed_loader
