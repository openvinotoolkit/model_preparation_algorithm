# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn

from mmcls.models.builder import CLASSIFIERS

from mpa.modules.models.classifiers.sam_classifier import ImageClassifier


@CLASSIFIERS.register_module()
class HybridClassifier(ImageClassifier):
    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(HybridClassifier, self).__init__(
            backbone, neck=neck, head=head, pretrained=pretrained
        )

    def forward_train(self, img, gt_label, **kwargs):
        img = [img[:, 0, :, :, :], img[:, 1, :, :, :]]
        img = torch.cat(img, dim=0)
        x = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return losses

    def extract_prob(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return nn.functional.softmax(self.head.fc(x)), x
