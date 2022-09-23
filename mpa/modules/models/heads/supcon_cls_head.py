# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torch import nn
import torch.nn.functional as F

from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.base_head import BaseHead


@HEADS.register_module()
class SupConClsHead(BaseHead):
    """Supervised Contrastive Loss for Classification Head

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from the backbone
        hid_channels (int): The channels of the hidden layer of the MLP
        out_channels (int): The channels of the output layer of the MLP
        loss (dict): configuration of loss, default is SupConLoss
        topk (set): evaluation topk score, default is (1, )
    """

    def __init__(self, num_classes, in_channels, hid_channels, out_channels=128, lamda=1.0, topk=(1, ),
                 loss=dict(type='SupConLoss', temperature=0.07, contrast_mode='all', base_temperature=0.07),
                 init_cfg=None):
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if hid_channels <= 0:
            raise ValueError(f"hid_channels={hid_channels} must be a positive integer")
        if out_channels <= 0:
            raise ValueError(f"out_channels={out_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")

        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        super(BaseHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        self.compute_loss = build_loss(loss)

        # Set up the standard classification head
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features=in_channels, out_features=self.num_classes)

        # Set up an MLP for the contrastive head
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hid_channels, out_features=out_channels),
        )

        self.lamda = lamda
        # self.loss = SupConLoss(temperature, contrast_mode, base_temperature)

    def forward_train(self, x, gt_labels, fc_only=False):
        """forward_train head using the Supervised Contrastive Loss

        Args:
            x (Tensor): features from the backbone.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses = dict()
        if self.fc is not None:
            fc_feats = self.fc(x)
            # behave like the cross entropy loss
            if fc_only:
                _, loss = self.compute_loss(None, gt_labels, fc_feats=fc_feats)
                losses.update(loss)
                return losses
        bsz = gt_labels.shape[0]
        mlp_feats = F.normalize(self.mlp(x), dim=1)
        f1, f2 = torch.split(mlp_feats, [bsz, bsz], dim=0)
        mlp_feats = torch.cat(f1.unsqueze(1), f2.unsqueeze(2), dim=1)
        loss_sc, loss_ce = self.compute_loss(mlp_feats, gt_labels, fc_feats=fc_feats)
        loss = loss_sc + self.lamda * loss_ce
        losses.update(loss)
        return losses
