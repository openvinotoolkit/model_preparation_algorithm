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
        logger.info(f'Semi-SL task config!!!!: training={training}')
        super().configure_task(cfg, training, **kwargs)

    def configure_task_cls_incr(self, cfg, task_adapt_type, org_model_classes, model_classes):
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
