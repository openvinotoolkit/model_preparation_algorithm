# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class ReduceMeanV1Attribute(Attribute):
    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceMeanV1(Operation):
    TYPE = "ReduceMean"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceMeanV1Attribute

    def forward(self, input, axes):
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return input

        return torch.mean(input=input, dim=axes, keepdim=self.attrs.keep_dims)
