# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from mmdet.core.anchor.builder import PRIOR_GENERATORS
from mmdet.core.anchor.anchor_generator import AnchorGenerator


@PRIOR_GENERATORS.register_module()
class SSDAnchorGeneratorClustered(AnchorGenerator):

    def __init__(self,
                 strides,
                 widths,
                 heights,
                 reclustering_anchors=False):
        self.strides = [_pair(stride) for stride in strides]
        self.widths = widths
        self.heights = heights
        self.centers = [(stride / 2., stride / 2.) for stride in strides]

        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for widths, heights, centers in zip(self.widths, self.heights, self.centers):
            base_anchors = self.gen_single_level_base_anchors(
                ws=torch.Tensor(widths),
                hs=torch.Tensor(heights),
                center=torch.Tensor(centers))
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      ws,
                                      hs,
                                      center):
        x_center, y_center = center

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors
