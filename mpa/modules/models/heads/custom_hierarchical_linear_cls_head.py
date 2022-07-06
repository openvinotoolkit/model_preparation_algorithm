# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmcls.models.builder import HEADS
from mmcls.models.heads import MultiLabelClsHead


@HEADS.register_module()
class CustomHierarchicalLinearClsHead(MultiLabelClsHead):
    """Custom Linear classification head for multilabel task.
    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        self.hierarchical_info = kwargs.pop('hierarchical_info', None)
        assert self.hierarchical_info
        super(CustomHierarchicalLinearClsHead, self).__init__(loss=loss)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self._init_layers()
        # build loss 두개 해야하네~

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        losses['loss'] = loss
        return losses

    def forward_train(self, x, gt_label, **kwargs): # TODO : 여기서 hierarchical info 받아서 파싱해야함.
        img_metas = kwargs.get('img_metas', False)
        gt_label = gt_label.type_as(x)
        cls_score = self.fc(x)
        
        for i in range(self.hierarchical_info['num_multiclass_heads']):
            head_gt = gt_label[:,i]
            head_logits = cls_score[:,self.hierarchical_info['head_idx_to_logits_range'][i][0] :
                                        self.hierarchical_info['head_idx_to_logits_range'][i][1]]
            valid_mask = head_gt >= 0
            head_gt = head_gt[valid_mask].long()
            head_logits = head_logits[valid_mask,:]

            if img_metas:
                valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
                losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
            else:
                losses = self.loss(cls_score, gt_label)

        if self.hierarchical_info['num_multiclass_heads'] > 1:
            losses /= self.hierarchical_info['num_multiclass_heads']

        # if self.multilabel_loss:
        #     head_gt = gt_label[:,self.hierarchical_info['num_multiclass_heads']:]
        #     head_logits = cls_score[:,self.hierarchical_info['num_single_label_classes']:]
        #     valid_mask = head_gt >= 0
        #     head_gt = head_gt[valid_mask].view(*valid_mask.shape)
        #     head_logits = head_logits[valid_mask].view(*valid_mask.shape)
        #     # multilabel_loss is assumed to perform no batch averaging
        #     losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)

        # if img_metas:
        #     valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
        #     losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        # else:
        #     losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, img):  # 여기도 필요하려나?
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.sigmoid(cls_score) if cls_score is not None else None
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
