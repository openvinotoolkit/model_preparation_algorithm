# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import LOSSES
from mmcls.models.losses import CrossEntropyLoss

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

@LOSSES.register_module()
class IBLoss(CrossEntropyLoss):
    def __init__(self, num_classes, start=5, weight=None, alpha=1000., **kwargs):
        super(IBLoss, self).__init__(loss_weight=1.0)
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.num_classes = num_classes
        self._cur_epoch = 0
        self._start_epoch = start
        print("IB LOSS")

    @property
    def cur_epoch(self):
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch):
        self._cur_epoch = epoch
        print(f"CUR EPOCH : {self._cur_epoch}")

    def forward(self, input, target, features):
        if self._cur_epoch < self._start_epoch:
            return super().forward(input, target)
        else:
            grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)),1) # N * 1
            ib = grads * features.reshape(-1)
            ib = self.alpha / (ib + self.epsilon)
            ce_loss = super().forward(input, target, reduction_override='none', weight=self.weight)
            loss = ce_loss * ib
            return loss.mean()

