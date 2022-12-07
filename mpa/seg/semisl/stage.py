# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv import ConfigDict
from mmseg.utils import get_root_logger
from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger
from mpa.seg.stage import SegStage
logger = get_logger()


class SemiSegStage(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # Model architecture
        if 'decode_head' in cfg.model:
            decode_head = cfg.model.decode_head
            if isinstance(decode_head, dict):
                decode_head.num_classes = len(model_classes)
            elif isinstance(decode_head, list):
                for head in decode_head:
                    head.num_classes = len(model_classes)

        return org_model_classes, model_classes, data_classes
