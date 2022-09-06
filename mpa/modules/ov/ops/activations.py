# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

from torch.nn import functional as F
import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class SoftMaxV0Attribute(Attribute):
    axis: int = field(default=1)


@OPS.register()
class SoftMaxV0(Operation):
    TYPE = "Softmax"
    VERSION = 0
    ATTRIBUTE_FACTORY = SoftMaxV0Attribute

    def forward(self, input):
        return F.softmax(input=input, dim=self.attrs.axis)


@dataclass
class SoftMaxV1Attribute(Attribute):
    axis: int = field(default=1)


@OPS.register()
class SoftMaxV1(Operation):
    TYPE = "Softmax"
    VERSION = 1
    ATTRIBUTE_FACTORY = SoftMaxV0Attribute

    def forward(self, input):
        return F.softmax(input=input, dim=self.attrs.axis)


@dataclass
class ReluV0Attribute(Attribute):
    pass


@OPS.register()
class ReluV0(Operation):
    TYPE = "Relu"
    VERSION = 0
    ATTRIBUTE_FACTORY = ReluV0Attribute

    def forward(self, input):
        return F.relu(input)


@dataclass
class SwishV4Attribute(Attribute):
    pass


@OPS.register()
class SwishV4(Operation):
    TYPE = "Swish"
    VERSION = 4
    ATTRIBUTE_FACTORY = SwishV4Attribute

    def forward(self, input, beta=1.0):
        return input * torch.sigmoid(input * beta)


@dataclass
class SigmoidV0Attribute(Attribute):
    pass


@OPS.register()
class SigmoidV0(Operation):
    TYPE = "Sigmoid"
    VERSION = 0
    ATTRIBUTE_FACTORY = SigmoidV0Attribute

    def forward(self, input):
        return torch.sigmoid(input)


@dataclass
class ClampV0Attribute(Attribute):
    min: float
    max: float


@OPS.register()
class ClampV0(Operation):
    TYPE = "Clamp"
    VERSION = 0
    ATTRIBUTE_FACTORY = ClampV0Attribute

    def forward(self, input):
        return input.clamp(min=self.attrs.min, max=self.attrs.max)


@dataclass
class PReluV0Attribute(Attribute):
    pass


@OPS.register()
class PReluV0(Operation):
    TYPE = "PRelu"
    VERSION = 0
    ATTRIBUTE_FACTORY = PReluV0Attribute

    def forward(self, input, slope):
        return F.prelu(input=input, weight=slope)


@dataclass
class TanhV0Attribute(Attribute):
    pass


@OPS.register()
class TanhV0(Operation):
    TYPE = "Tanh"
    VERSION = 0
    ATTRIBUTE_FACTORY = TanhV0Attribute

    def forward(self, input):
        return F.tanh(input)


@dataclass
class EluV0Attribute(Attribute):
    alpha: float


@OPS.register()
class EluV0(Operation):
    TYPE = "Elu"
    VERSION = 0
    ATTRIBUTE_FACTORY = EluV0Attribute

    def forward(self, input):
        return F.elu(input=input, alpha=self.attrs.alpha)


@dataclass
class SeluV0Attribute(Attribute):
    pass


@OPS.register()
class SeluV0(Operation):
    TYPE = "Selu"
    VERSION = 0
    ATTRIBUTE_FACTORY = SeluV0Attribute

    def forward(self, input, alpha, lambda_):
        return lambda_ * F.elu(input=input, alpha=alpha)


@dataclass
class MishV4Attribute(Attribute):
    pass


@OPS.register()
class MishV4(Operation):
    TYPE = "Mish"
    VERSION = 4
    ATTRIBUTE_FACTORY = MishV4Attribute

    def forward(self, input):
        return F.mish(input=input)


@dataclass
class HSwishV4Attribute(Attribute):
    pass


@OPS.register()
class HSwishV4(Operation):
    TYPE = "HSwish"
    VERSION = 4
    ATTRIBUTE_FACTORY = HSwishV4Attribute

    def forward(self, input):
        return F.hardswish(input=input)


@dataclass
class HSigmoidV5Attribute(Attribute):
    pass


@OPS.register()
class HSigmoidV5(Operation):
    TYPE = "HSigmoid"
    VERSION = 5
    ATTRIBUTE_FACTORY = HSigmoidV5Attribute

    def forward(self, input):
        return F.hardsigmoid(input=input)
