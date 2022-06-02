# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
from mmdet.models import LOSSES
from mmdet.models.losses.focal_loss import sigmoid_focal_loss
from mmdet.models.losses.varifocal_loss import varifocal_loss


def cross_sigmoid_focal_loss(inputs,
                             targets,
                             weight=None,
                             num_classes=None,
                             alpha=0.25,
                             gamma=2,
                             reduction="mean",
                             avg_factor=None,
                             use_vfl=False,
                             ignored_masks=None):
    """
    Arguments:
       - inputs: inputs Tensor (N * C)
       - targets: targets Tensor (N), if use_vfl, then Tensor (N * C)
       - weights: weights Tensor (N), consists of (binarized label schema * weights)
       - num_classes: number of classes for training
       - alpha: focal loss alpha
       - gamma: focal loss gamma
       - reduction: default = mean
       - avg_factor: average factors
    """
    cross_mask = inputs.new_ones(inputs.shape, dtype=torch.int8)
    if ignored_masks is not None:
        neg_mask = targets.sum(axis=1) == 0 if use_vfl else targets == num_classes
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]
        cross_mask[neg_idx] = ignored_masks[neg_idx].type(torch.int8)

    if use_vfl:
        loss = varifocal_loss(inputs, targets,
                              weight=weight,
                              gamma=gamma,
                              alpha=alpha,
                              reduction='none',
                              avg_factor=None) * cross_mask
    else:
        loss = sigmoid_focal_loss(inputs, targets,
                                  weight=weight,
                                  gamma=gamma,
                                  alpha=alpha,
                                  reduction='none',
                                  avg_factor=None) * cross_mask

    if reduction == "mean":
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@LOSSES.register_module()
class CrossSigmoidFocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 num_classes=None,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=None):
        super(CrossSigmoidFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid

        self.cls_criterion = cross_sigmoid_focal_loss

    def forward(self,
                pred,
                targets,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                use_vfl=False,
                ignored_masks=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            pred,
            targets,
            weight=weight,
            num_classes=self.num_classes,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor,
            use_vfl=use_vfl,
            ignored_masks=ignored_masks
            )
        return loss_cls
