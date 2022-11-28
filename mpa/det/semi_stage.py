# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mpa.det.incr_stage import IncrDetectionStage
from mpa.utils.logger import get_logger

logger = get_logger()


class SemiDetectionStage(IncrDetectionStage):
    """Patch config to support semi supervised learning for object detection"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        if 'task_adapt' in cfg:
            logger.info(f'task config!!!!: training={training}')
            task_adapt_type = cfg['task_adapt'].get('type', None)
            task_adapt_op = cfg['task_adapt'].get('op', 'REPLACE')

            org_model_classes, model_classes, data_classes = \
                self.configure_classes(cfg, task_adapt_type, task_adapt_op)
            if data_classes != model_classes:
                self.configure_task_data_pipeline(cfg, model_classes, data_classes)
            # TODO[JAEGUK]: configure_anchor is not working
            if cfg['task_adapt'].get('use_mpa_anchor', False):
                self.configure_anchor(cfg)
            self.configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes)
            self.configure_task_semi()

    def configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes):
        """Patch for class incremental learning.
        Semi supervised learning should support incrmental learning
        """
        if task_adapt_type == 'mpa':
            self.configure_bbox_head(cfg, model_classes)
            self.configure_task_adapt_hook(cfg, org_model_classes, model_classes)
            self.configure_val_interval(cfg)
        else:
            src_data_cfg = self.get_train_data_cfg(cfg)
            src_data_cfg.pop('old_new_indices', None)
