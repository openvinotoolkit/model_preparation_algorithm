# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmcls.models.builder import HEADS, build_loss
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
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 multilabel_loss=dict(
                     type='AsymmetricLoss',
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        self.hierarchical_info = kwargs.pop('hierarchical_info', None)
        assert self.hierarchical_info
        super(CustomHierarchicalLinearClsHead, self).__init__(loss=loss)
        if self.hierarchical_info['num_multilabel_classes'] > 0:
            self.compute_multilabel_loss = build_loss(multilabel_loss)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, cls_score, gt_label, multilabel=False, valid_label_mask=None):
        num_samples = len(cls_score)
        # compute loss
        if multilabel:
            gt_label = gt_label.type_as(cls_score)
            # map difficult examples to positive ones
            _gt_label = torch.abs(gt_label)

            loss = self.compute_multilabel_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        else:
            loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)

        return loss

    def forward_train(self, x, gt_label, **kwargs):
        img_metas = kwargs.get('img_metas', False)
        gt_label = gt_label.type_as(x)
        cls_score = self.fc(x)
        
        losses = dict(loss=0.)
        for i in range(self.hierarchical_info['num_multiclass_heads']):
            head_gt = gt_label[:,i]
            head_logits = cls_score[:,self.hierarchical_info['head_idx_to_logits_range'][i][0] :
                                      self.hierarchical_info['head_idx_to_logits_range'][i][1]]
            valid_mask = head_gt >= 0
            head_gt = head_gt[valid_mask].long()
            head_logits = head_logits[valid_mask,:]
            multiclass_loss = self.loss(head_logits, head_gt)
            losses['loss'] += multiclass_loss
            print(f'multiclass: {multiclass_loss}, losses: {losses}')

        if self.hierarchical_info['num_multiclass_heads'] > 1:
            losses['loss'] /= self.hierarchical_info['num_multiclass_heads']

        if self.compute_multilabel_loss:
            head_gt = gt_label[:,self.hierarchical_info['num_multiclass_heads']:]
            head_logits = cls_score[:,self.hierarchical_info['num_single_label_classes']:]
            valid_mask = head_gt >= 0
            head_gt = head_gt[valid_mask].view(*valid_mask.shape)
            head_logits = head_logits[valid_mask].view(*valid_mask.shape)

            # multilabel_loss is assumed to perform no batch averaging
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            multilabel_loss = self.loss(head_logits, head_gt, multilabel=True, valid_label_mask=valid_label_mask)
            losses['loss'] += multilabel_loss.mean()
            print(f'multilabel: {multilabel_loss.mean()}, losses: {losses}')
        print('forward_train end!! ')
        return losses

    def simple_test(self, img):  # 여기도 필요하려나? 여기도 위에 heat_gt랑 logit 파싱하는부분이필요하네....
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # for i in range(self.hierarchical_info['num_multiclass_heads']): # 이거 어떻게하지..
        multiclass_logits = cls_score[:,self.hierarchical_info['head_idx_to_logits_range'][0][0] :
                                        self.hierarchical_info['head_idx_to_logits_range'][0][1]]
        multilabel_logits = cls_score[:,self.hierarchical_info['num_single_label_classes']:]
        multiclass_pred = F.softmax(multiclass_logits) if multiclass_logits is not None else None
        multilabel_pred = F.sigmoid(multilabel_logits) if multilabel_logits is not None else None
        pred = torch.cat([multiclass_pred, multilabel_pred], axis=1)
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
