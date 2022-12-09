# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv import ConfigDict
from mmseg.utils import get_root_logger
from mpa.stage import Stage
from mpa.utils.logger import get_logger

logger = get_logger()


class SegStage(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure!: training={training}')

        cfg = self.cfg
        self.configure_model(cfg, model_cfg, training, **kwargs)
        self.configure_ckpt(cfg, model_ckpt, kwargs.get('pretrained', None))
        self.configure_data(cfg, data_cfg, training)
        self.configure_task(cfg, training, **kwargs)
        self.configure_hyperparams(cfg, training, **kwargs)

        return cfg

    def configure_model(self, cfg, model_cfg, training, **kwargs):
        if model_cfg:
            if hasattr(model_cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                raise ValueError("Unexpected config was passed through 'model_cfg'. "
                                 "it should have 'model' attribute in the config")
            cfg.model_task = cfg.model.pop('task', 'segmentation')
            if cfg.model_task != 'segmentation':
                raise ValueError(
                    f'Given model_cfg ({model_cfg.filename}) is not supported by segmentation recipe'
                )

    def configure_ckpt(self, cfg, model_ckpt, pretrained=None):
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained  # Overriding by stage input

        if cfg.get('resume', False):
            cfg.resume_from = cfg.load_from

    def configure_data(self, cfg, data_cfg, training):
        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        if training:
            if cfg.data.get('val', False):
                self.validate = True
        # Dataset
        src_data_cfg = Stage.get_train_data_cfg(cfg)
        for mode in ['train', 'val', 'test']:
            if src_data_cfg.type == 'MPASegDataset':
                if cfg.data[mode]['type'] != 'MPASegDataset':
                    # Wrap original dataset config
                    org_type = cfg.data[mode]['type']
                    cfg.data[mode]['type'] = 'MPASegDataset'
                    cfg.data[mode]['org_type'] = org_type

    def configure_hyperparams(self, cfg, training, **kwargs):
        if 'hyperparams' in cfg:
            hyperparams = kwargs.get('hyperparams', None)
            if hyperparams is not None:
                bs = hyperparams.get('bs', None)
                if bs is not None:
                    cfg.data.samples_per_gpu = bs

                lr = hyperparams.get('lr', None)
                if lr is not None:
                    cfg.optimizer.lr = lr
