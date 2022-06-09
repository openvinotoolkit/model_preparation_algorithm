# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy, LossEqualizer
from mmseg.ops import resize
from mmseg.core import add_prefix
from mpa.modules.utils.mask_utils import get_ignored_labels_per_batch

@HEADS.register_module()
class CustomFCNHead(FCNHead):
    """Custom Fully Convolution Networks for Semantic Segmentation.
    """

    def __init__(self,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 align_corners=False,
                 enable_loss_equalizer=False,
                 **kwargs):
        super(CustomFCNHead, self).__init__(**kwargs)

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.enable_loss_equalizer = enable_loss_equalizer

        loss_configs = loss_decode if isinstance(loss_decode, (tuple, list)) else [loss_decode]
        assert len(loss_configs) > 0
        self.loss_modules = nn.ModuleList([
            build_loss(loss_cfg, self.ignore_index)
            for loss_cfg in loss_configs
        ])

        self.loss_equalizer = None
        if enable_loss_equalizer:
            self.loss_equalizer = LossEqualizer()

        self.forward_output = None

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, pixel_weights=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', 'img_norm_cfg',
                and 'ignored_labels'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        ignored_labels = get_ignored_labels_per_batch(img_metas, self.num_classes)
        losses = self.losses(seg_logits, gt_semantic_seg, ignored_labels, train_cfg, pixel_weights)

        if self.forward_output is not None:
            return losses, self.forward_output
        else:
            return losses, seg_logits

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, ignored_masks, train_cfg, pixel_weights=None):
        """Compute segmentation loss."""

        loss = dict()

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        seg_label = seg_label.squeeze(1)
        out_losses = dict()
        for loss_idx, loss_module in enumerate(self.loss_modules):
            loss_value, loss_meta = loss_module(
                seg_logit,
                seg_label,
                ignored_masks,
                pixel_weights=pixel_weights
            )

            loss_name = loss_module.name + f'-{loss_idx}'
            out_losses[loss_name] = loss_value
            loss.update(add_prefix(loss_meta, loss_name))

        if self.enable_loss_equalizer and len(self.loss_modules) > 1:
            out_losses = self.loss_equalizer.reweight(out_losses)

        for loss_name, loss_value in out_losses.items():
            loss[loss_name] = loss_value

        loss['loss_seg'] = sum(out_losses.values())
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        if train_cfg.mix_loss.enable:
            mix_loss = self._mix_loss(
                seg_logit,
                seg_label,
                ignore_index=self.ignore_index
            )

            mix_loss_weight = train_cfg.mix_loss.get('weight', 1.0)
            loss['loss_mix'] = mix_loss_weight * mix_loss

        return loss
