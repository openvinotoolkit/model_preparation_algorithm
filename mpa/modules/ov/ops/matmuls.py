# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class MatMulV0Attribute(Attribute):
    transpose_a: bool = field(default=False)
    transpose_b: bool = field(default=False)


@OPS.register()
class MatMulV0(Operation):
    TYPE = "MatMul"
    VERSION = 0
    ATTRIBUTE_FACTORY = MatMulV0Attribute

    def forward(self, input_a, input_b):
        if self.attrs.transpose_a:
            input_a = torch.transpose(input_a, -1, -2)
        if self.attrs.transpose_b:
            input_b = torch.transpose(input_b, -1, -2)
        return torch.matmul(input_a, input_b)
