# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmcls.models.builder import HEADS
from mmcls.models.heads import MultiLabelClsHead


@HEADS.register_module()
class CustomMultiLabelGroupFCClsHead(MultiLabelClsHead):
    """Custom GroupFC classification head for multilabel task.
    Args:
        num_classes (int): Number of categories.
        in_groups (int): Number of input embedding groups.
        in_embedding_size (int): Size of input embedding.
        normalized (bool): Normalize input embeddings and weights.
        scale (float): positive scale parameter.
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 in_groups,
                 in_embedding_size,
                 normalized=False,
                 scale=1.0,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        super().__init__(loss=loss)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.num_classes = num_classes
        self.embed_len_decoder = min(num_classes, in_groups)
        self.in_groups = in_groups
        self.normalized = normalized
        self.scale = scale
        self.in_embedding_size = in_embedding_size
        self.duplicate_factor = math.ceil(self.num_classes / self.embed_len_decoder)
        self._init_layers()

    def _init_layers(self):
        self.duplicate_pooling = nn.Parameter(
            torch.Tensor(self.embed_len_decoder, self.in_embedding_size, self.duplicate_factor))
        self.duplicate_pooling_bias = nn.Parameter(torch.Tensor(self.num_classes))

        self.group_fc = GroupFC(self.normalized)

    def init_weights(self):
        nn.init.xavier_normal_(self.duplicate_pooling)
        nn.init.constant_(self.duplicate_pooling_bias, 0.)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        losses['loss'] = loss / self.scale
        return losses

    def forward_group_fc(self, h):
        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.duplicate_factor, device=h.device, dtype=h.dtype)
        self.group_fc(h, self.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, : self.num_classes]
        h_out += self.duplicate_pooling_bias
        return h_out

    def forward_train(self, x, gt_label, **kwargs):
        img_metas = kwargs.get('img_metas', False)
        gt_label = gt_label.type_as(x)

        cls_score = self.forward_group_fc(x) * self.scale
        if img_metas:
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        else:
            losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.forward_group_fc(img) * self.scale
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = torch.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if 'ignored_labels' in meta and meta['ignored_labels']:
                mask[meta['ignored_labels']] = 0
            mask = mask.cuda() if torch.cuda.is_available() else mask
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask


class GroupFC(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def __call__(self, h, duplicate_pooling, out_extrap):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape) == 3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            if self.normalize:
                h_i = F.normalize(h_i.view(h_i.shape[0], -1), dim=1)
                w_i = F.normalize(w_i, p=2., dim=0)
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)