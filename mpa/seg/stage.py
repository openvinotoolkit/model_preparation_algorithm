# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv import ConfigDict
from mmseg.utils import get_root_logger
from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger

logger = get_logger()


class SegStage(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure!: training={training}')

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(model_cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                raise ValueError("Unexpected config was passed through 'model_cfg'. "
                                 "it should have 'model' attribute in the config")
            model_task = cfg.model.pop('task', 'segmentation')
            if model_task != 'segmentation':
                raise ValueError(
                    f'Given model_cfg ({model_cfg.filename}) is not supported by segmentation recipe'
                )
        self.configure_model(cfg, training, **kwargs)

        if not cfg.get('task_adapt'):   # if task_adapt dict is empty(semi-sl), just pop to pass task_adapt
            cfg.pop('task_adapt')

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained  # Overriding by stage input

        if cfg.get('resume', False):
            cfg.resume_from = cfg.load_from

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)

        if training:
            if cfg.data.get('val', False):
                self.validate = True

        # Task
        if 'task_adapt' in cfg:
            self.configure_task(cfg, training, **kwargs)

        # Other hyper-parameters
        if 'hyperparams' in cfg:
            self.configure_hyperparams(cfg, training, **kwargs)

        return cfg

    def configure_model(self, cfg, training, **kwargs):
        pass

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation
        """
        self.logger = get_root_logger()
        self.logger.info(f'task config!!!!: training={training}')
        task_adapt_type = cfg['task_adapt'].get('type', None)
        task_adapt_op = cfg['task_adapt'].get('op', 'REPLACE')

        # Task classes
        org_model_classes, model_classes, data_classes = \
            self.configure_task_classes(cfg, task_adapt_op)

        # Incremental learning
        if task_adapt_type == 'mpa':
            self.configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes)

    def configure_cross_entropy_loss_with_ignore(self, model_classes):
        cfg_loss_decode = ConfigDict(
            type='CrossEntropyLossWithIgnore',
            reduction='mean',
            sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
            loss_weight=1.0
        )
        return cfg_loss_decode

    def configure_am_softmax_loss_with_ignore(self, model_classes):
        cfg_loss_decode = ConfigDict(
            type='AMSoftmaxLossWithIgnore',
            scale_cfg=ConfigDict(
                type='PolyScalarScheduler',
                start_scale=30,
                end_scale=5,
                by_epoch=True,
                num_iters=250,
                power=1.2
            ),
            margin_type='cos',
            margin=0.5,
            gamma=0.0,
            t=1.0,
            target_loss='ce',
            pr_product=False,
            conf_penalty_weight=ConfigDict(
                type='PolyScalarScheduler',
                start_scale=0.2,
                end_scale=0.15,
                by_epoch=True,
                num_iters=200,
                power=1.2
            ),
            border_reweighting=False,
            sampler=ConfigDict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
            loss_weight=1.0
        )
        return cfg_loss_decode

    def configure_task_cls_incr(self, cfg, task_adapt_type, org_model_classes, model_classes):
        new_classes = np.setdiff1d(model_classes, org_model_classes).tolist()
        has_new_class = True if len(new_classes) > 0 else False
        if has_new_class is False:
            ValueError('Incremental learning should have at least one new class!')

        # Incremental Learning
        if cfg.get('ignore', False):
            if 'decode_head' in cfg.model:
                decode_head = cfg.model.decode_head
                if isinstance(decode_head, dict):
                    if decode_head.type == 'FCNHead':
                        decode_head.type = 'CustomFCNHead'
                        decode_head.loss_decode = self.configure_cross_entropy_loss_with_ignore(model_classes)
                    elif decode_head.type == 'OCRHead':
                        decode_head.type = 'CustomOCRHead'
                        decode_head.loss_decode = self.configure_am_softmax_loss_with_ignore(model_classes)
                elif isinstance(decode_head, list):
                    for head in decode_head:
                        if head.type == 'FCNHead':
                            head.type = 'CustomFCNHead'
                            head.loss_decode = [self.configure_cross_entropy_loss_with_ignore(model_classes)]
                        elif head.type == 'OCRHead':
                            head.type = 'CustomOCRHead'
                            head.loss_decode = [self.configure_am_softmax_loss_with_ignore(model_classes)]

        # Dataset
        src_data_cfg = Stage.get_train_data_cfg(cfg)
        for mode in ['train', 'val', 'test']:
            if src_data_cfg.type == 'MPASegDataset':
                if cfg.data[mode]['type'] != 'MPASegDataset':
                    # Wrap original dataset config
                    org_type = cfg.data[mode]['type']
                    cfg.data[mode]['type'] = 'MPASegDataset'
                    cfg.data[mode]['org_type'] = org_type
            else:
                if cfg.data[mode]['type'] == 'SegTaskAdaptDataset':
                    cfg.data[mode]['classes'] = model_classes
                    if has_new_class:
                        cfg.data[mode]['new_classes'] = new_classes
                    cfg.data[mode]['with_background'] = True
                else:
                    # Wrap original dataset config
                    org_type = cfg.data[mode]['type']
                    cfg.data[mode]['type'] = 'SegTaskAdaptDataset'
                    cfg.data[mode]['org_type'] = org_type
                    cfg.data[mode]['classes'] = model_classes
                    if has_new_class:
                        cfg.data[mode]['new_classes'] = new_classes
                    cfg.data[mode]['with_background'] = True

        # Update Task Adapt Hook
        task_adapt_hook = ConfigDict(
            type='TaskAdaptHook',
            src_classes=org_model_classes,
            dst_classes=model_classes,
            model_type=cfg.model.type,
            sampler_flag=has_new_class,
            efficient_mode=cfg['task_adapt'].get('efficient_mode', False)
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_task_classes(self, cfg, task_adapt_op):
        # Task classes
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)
        if 'background' not in org_model_classes:
            org_model_classes = ['background'] + org_model_classes
        if 'background' not in data_classes:
            data_classes = ['background'] + data_classes

        # Model classes
        if task_adapt_op == 'REPLACE':
            if len(data_classes) == 1: # 'background'
                model_classes = org_model_classes.copy()
            else:
                model_classes = data_classes.copy()
        elif task_adapt_op == 'MERGE':
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f'{task_adapt_op} is not supported for task_adapt options!')

        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        # Model architecture
        if 'decode_head' in cfg.model:
            decode_head = cfg.model.decode_head
            if isinstance(decode_head, dict):
                decode_head.num_classes = len(model_classes)
            elif isinstance(decode_head, list):
                for head in decode_head:
                    head.num_classes = len(model_classes)

        return org_model_classes, model_classes, data_classes

    def configure_hyperparams(self, cfg, training, **kwargs):
        hyperparams = kwargs.get('hyperparams', None)
        if hyperparams is not None:
            bs = hyperparams.get('bs', None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get('lr', None)
            if lr is not None:
                cfg.optimizer.lr = lr
