# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook

from mpa.utils.logger import get_logger

logger = get_logger()

@HOOKS.register_module()
class NorCalHook(Hook):
    """
    NorCal hook for post calibration and normalization
    Paper: https://proceedings.neurips.cc/paper/2021/file/14ad095ecc1c3e1b87f3c522836e9158-Paper.pdf

    TODO: explain Args
    Args:
        gamma: hyperparameter for determining penalty
    """

    def __init__(self,
                 gamma,):
        self.gamma = gamma

    def before_train_epoch(self, runner):
        cls_distribution = self.get_cls_distribution(runner.data_loader.dataset)
        calib_scale = self.gamma * torch.Tensor(cls_distribution).log()
        logger.info(f'NorCal calibration scale: {calib_scale}')
        breakpoint()

    def get_cls_distribution(self, dataset):
        breakpoint()
        return [len(dataset.img_indices[data_cls]) for data_cls in self.dst_classes]


