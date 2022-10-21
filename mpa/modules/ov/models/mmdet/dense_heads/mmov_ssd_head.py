# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.ssd_head import SSDHead

from ...mmov_model import MMOVModel


@HEADS.register_module()
class MMOVSSDHead(SSDHead):
    def __init__(
        self,
        model_path: str,
        weight_path: Optional[str] = None,
        inputs: Optional[
            Union[Dict[str, Union[str, List[str]]], List[str], str]
        ] = None,
        outputs: Optional[
            Union[Dict[str, Union[str, List[str]]], List[str], str]
        ] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        *args,
        **kwargs,
    ):

        self._model_path = model_path
        self._weight_path = weight_path
        self._inputs = deepcopy(inputs)
        self._outputs = deepcopy(outputs)
        self._init_weight = init_weight
        self._verify_shape = verify_shape

        # dummy input
        in_channels = (512, 1024, 512, 256, 256, 256)
        super().__init__(in_channels=in_channels, *args, **kwargs)

        self.cls_convs = torch.nn.ModuleList()
        self.reg_convs = torch.nn.ModuleList()

        for (
            inputs,
            outputs,
        ) in zip(self._inputs["cls_convs"], self._outputs["cls_convs"]):
            self.cls_convs.append(
                MMOVModel(
                    self._model_path,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

        for (
            inputs,
            outputs,
        ) in zip(self._inputs["reg_convs"], self._outputs["reg_convs"]):
            self.reg_convs.append(
                MMOVModel(
                    self._model_path,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(
            feats, self.reg_convs, self.cls_convs
        ):

            batch_dim = feat.size(0)
            # since mmdet v2.0, SSDHead is supposed to be
            # that FG labels to [0, num_class-1] and BG labels to num_class
            # but ssd300, ssd512 from OMZ are
            # that FG labels to [1, num_class] and BG labels to 0
            cls_score = cls_conv(feat)
            cls_score = cls_score.permute(0, 2, 3, 1)
            shape = cls_score.shape
            cls_score = cls_score.reshape(batch_dim, -1, self.cls_out_channels)
            cls_score = torch.cat((cls_score[:, :, 1:], cls_score[:, :, 0:1]), -1)
            cls_score = cls_score.reshape(*shape)
            cls_score = cls_score.permute(0, 3, 1, 2)

            bbox_pred = reg_conv(feat)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds
