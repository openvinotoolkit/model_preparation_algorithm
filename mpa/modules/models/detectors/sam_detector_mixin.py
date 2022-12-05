# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from mmdet.models.detectors import BaseDetector, SingleStageDetector
from mmdet.utils.deployment.export_helpers import get_feature_vector, get_saliency_map
from mmdet.integration.nncf.utils import no_nncf_trace
from mpa.modules.hooks.auxiliary_hooks import DetSaliencyMapHook


class SAMDetectorMixin(BaseDetector):
    """SAM-enabled detector mix-in
    """
    def train_step(self, data, optimizer, **kwargs):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data
        return super().train_step(data, optimizer, **kwargs)

    def simple_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    postprocess=True):

        """
        Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        """
        x = self.extract_feat(img)
        if isinstance(self, SingleStageDetector):
            outs = self.bbox_head(x)
            with no_nncf_trace():
                bbox_results = \
                    self.bbox_head.get_bboxes(*outs, img_metas, self.test_cfg, False)
                if torch.onnx.is_in_onnx_export():
                    feature_vector = get_feature_vector(x)
                    cls_scores = outs[0]
                    saliency_map = DetSaliencyMapHook(self).func(cls_scores, cls_scores_provided=True)
                    feature = feature_vector, saliency_map
                    return bbox_results[0], feature

            if postprocess:
                bbox_results = [
                    self.postprocess(det_bboxes, det_labels, None, img_metas, rescale=rescale)
                    for det_bboxes, det_labels in bbox_results
                ]
            return bbox_results

        else:
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            out = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale, postprocess=postprocess)
            with no_nncf_trace():
                if torch.onnx.is_in_onnx_export():
                    feature_vector = get_feature_vector(x)
                    saliency_map = get_saliency_map(x[-1])
                    feature = feature_vector, saliency_map
                    return out, feature
            return out
