# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn

from mmcls.models.builder import CLASSIFIERS

from mpa.modules.models.classifiers.sam_classifier import ImageClassifier


@CLASSIFIERS.register_module()
class SupConClassifier(ImageClassifier):
    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(SupConClassifier, self).__init__(
            backbone, neck=neck, head=head, pretrained=pretrained
        )

    # def train_step(self, data, optimizer):
    #     data['img_1'], data['img_2'] = data['img'][:,0,...], data['img'][:,1,...]
    #     self.current_batch = data
    #     return super().train_step(data, optimizer)

    def forward_train(self, img, gt_label, **kwargs):
        # img = img.reshape((-1, img.shape[2], img.shape[3], img.shape[4]))
        img = [img[:, 0, :, :, :], img[:, 1, :, :, :]]
        img = torch.cat((img[0], img[1]), dim=0)
        x = self.extract_feat(img)
        losses = dict()
        loss = self.head.forward_train(x, gt_label, fc_only=False)
        losses.update(loss)
        return losses

    def extract_prob(self, img):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return nn.functional.softmax(self.head.fc(x)), x
