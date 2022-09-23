# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
import numpy as np

from mmcls.models.builder import LOSSES
from mmcls.models.losses import CrossEntropyLoss


def ib_loss(input_values, ib):
    """Computes the ib loss"""
    loss = input_values * ib
    return loss.mean()


@LOSSES.register_module()
class IBLoss(CrossEntropyLoss):
    def __init__(self, num_classes, start=5, cls_num_list=None, alpha=1000., **kwargs):
        super(IBLoss, self).__init__(loss_weight=1.0)
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = self.get_weight(cls_num_list)
        self.num_classes = num_classes
        self._cur_epoch = 0
        self._start_epoch = start

    def get_weight(self, cls_num_list):
        if cls_num_list is None:
            return None
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        return per_cls_weights

    @property
    def cur_epoch(self):
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch):
        self._cur_epoch = epoch

    def forward(self, input, target, features):
        if self._cur_epoch < self._start_epoch:
            return super().forward(input, target)
        else:
            grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)), 1)
            ib = grads * features.reshape(-1)
            ib = self.alpha / (ib + self.epsilon)
            ce_loss = F.cross_entropy(input, target, weight=self.weight.to(input.get_device()), reduction='none')
            loss = ce_loss * ib
            return loss.mean()
