# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torch.nn.functional import softmax
from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier


@CLASSIFIERS.register_module()
class SupConClassifier(ImageClassifier):
    def __init__(self, backbone=None, neck=None, head=None, pretrained=None, **kwargs):
        super(SupConClassifier, self).__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
        )
        self.hierarchical = False
        self.multilabel = False

    def forward_train(self, img, gt_label, **kwargs):
        # concatenate the different image views along the batch size
        if len(img.shape) == 5:
            img = torch.cat([img[:, d, :, :, :] for d in range(img.shape[1])], dim=0)
        x = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return losses

    def extract_prob(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return softmax(self.head.fc(x)), x
