# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import torch
from mmdet.models.builder import NECKS
from mmdet.models.necks.ssd_neck import SSDNeck

from ...mmov_model import MMOVModel


@NECKS.register_module()
class MMOVSSDNeck(SSDNeck):
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
    ):
        # dummy
        in_channels = (512, 1024)
        out_channels = (512, 1024, 512, 256, 256, 256)
        level_strides = (2, 2, 1, 1)
        level_paddings = (1, 1, 0, 0)
        l2_norm_scale = None
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            level_strides=level_strides,
            level_paddings=level_paddings,
            l2_norm_scale=l2_norm_scale,
        )

        self._model_path = model_path
        self._weight_path = weight_path
        self._init_weight = init_weight

        self.extra_layers = torch.nn.ModuleList()

        for input, output in zip(inputs["extra_layers"], outputs["extra_layers"]):
            if input and output:
                self.extra_layers.append(
                    MMOVModel(
                        model_path,
                        weight_path,
                        inputs=input,
                        outputs=output,
                        remove_normalize=False,
                        merge_bn=False,
                        paired_bn=False,
                        init_weight=init_weight,
                        verify_shape=verify_shape,
                    )
                )
            else:
                self.extra_layers.append(torch.nn.Identity())

        if "l2_norm" in inputs and "l2_norm" in outputs:
            for input, output in zip(inputs["l2_norm"], outputs["l2_norm"]):
                self.l2_norm = MMOVModel(
                    model_path,
                    weight_path,
                    inputs=input,
                    outputs=output,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=init_weight,
                    verify_shape=verify_shape,
                )

    def init_weights(self, pretrained=None):
        # TODO
        pass
