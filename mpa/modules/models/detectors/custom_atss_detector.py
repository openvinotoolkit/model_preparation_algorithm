# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.atss import ATSS
from .sam_detector_mixin import SAMDetectorMixin
from .l2sp_detector_mixin import L2SPDetectorMixin
from mpa.modules.utils.task_adapt import map_class_names
from mpa.utils.logger import get_logger
from mpa.deploy.utils import is_mmdeploy_enabled

logger = get_logger()


@DETECTORS.register_module()
class CustomATSS(SAMDetectorMixin, L2SPDetectorMixin, ATSS):
    """SAM optimizer & L2SP regularizer enabled custom ATSS
    """
    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt['dst_classes'],  # model_classes
                    task_adapt['src_classes']   # chkpt_classes
                )
            )

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        return super().forward_train(
              img,
              img_metas,
              gt_bboxes,
              gt_labels,
              gt_bboxes_ignore=gt_bboxes_ignore
        )

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading
        """
        logger.info(f'----------------- CustomATSS.load_state_dict_pre_hook() called w/ prefix: {prefix}')

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f'{chkpt_classes} -> {model_classes} ({model2chkpt})')

        model_dict = model.state_dict()
        param_names = [
            'bbox_head.atss_cls.weight',
            'bbox_head.atss_cls.bias',
        ]
        for model_name in param_names:
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f'Skipping weight copy: {chkpt_name}')
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for m, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[m].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER
    from mpa.modules.hooks.recording_forward_hooks import (
        FeatureVectorHook,
        ActivationMapHook,
    )

    @FUNCTION_REWRITER.register_rewriter(
        "mpa.modules.models.detectors.custom_atss_detector."
        "CustomATSS.simple_test"
    )
    def custom_atss__simple_test(ctx, self, img, img_metas, **kwargs):
        feat = self.extract_feat(img)
        out = self.bbox_head.simple_test(feat, img_metas, **kwargs)
        feature_vector = FeatureVectorHook.func(feat)
        sailency_map = ActivationMapHook.func(feat[-1])
        return (*out, feature_vector, sailency_map)
