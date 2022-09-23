# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead
from mpa.modules.models.heads.non_linear_cls_head import NonLinearClsHead


@HEADS.register_module()
class IBLossHead(LinearClsHead):
    def __init__(self, **kwargs):
        super(IBLossHead, self).__init__(**kwargs)

    def forward_train(self, x, gt_label):
        cls_score = self.fc(x)
        losses = dict()
        loss = self.compute_loss(cls_score, gt_label, torch.sum(torch.abs(x), 1).reshape(-1, 1))
        losses['loss'] = loss
        return losses


@HEADS.register_module()
class NonLinearIBLossHead(NonLinearClsHead):
    def __init__(self, **kwargs):
        super(NonLinearIBLossHead, self).__init__(**kwargs)

    def forward_train(self, x, gt_label):
        cls_score = self.classifier(x)
        losses = dict()
        loss = self.compute_loss(cls_score, gt_label, torch.sum(torch.abs(x), 1).reshape(-1, 1))
        losses['loss'] = loss
        return losses
