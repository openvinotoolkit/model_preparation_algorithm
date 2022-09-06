# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class MultiplyV1Attribute(Attribute):
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class MultiplyV1(Operation):
    TYPE = "Multiply"
    VERSION = 1
    ATTRIBUTE_FACTORY = MultiplyV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 * input_1
        elif broadcast == "numpy":
            return input_0 * input_1
        else:
            raise NotImplementedError


@dataclass
class DivideV1Attribute(Attribute):
    m_pythondiv: bool = field(default=True)
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class DivideV1(Operation):
    TYPE = "Divide"
    VERSION = 1
    ATTRIBUTE_FACTORY = DivideV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 / input_1
        elif broadcast == "numpy":
            return input_0 / input_1
        else:
            raise NotImplementedError


@dataclass
class AddV1Attribute(Attribute):
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class AddV1(Operation):
    TYPE = "Add"
    VERSION = 1
    ATTRIBUTE_FACTORY = AddV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 + input_1
        elif broadcast == "numpy":
            return input_0 + input_1
        else:
            raise NotImplementedError


@dataclass
class SubtractV1Attribute(Attribute):
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class SubtractV1(Operation):
    TYPE = "Subtract"
    VERSION = 1
    ATTRIBUTE_FACTORY = SubtractV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 - input_1
        elif broadcast == "numpy":
            return input_0 - input_1
        else:
            raise NotImplementedError


@dataclass
class TanV0Attribute(Attribute):
    pass


@OPS.register()
class TanV0(Operation):
    TYPE = "Tan"
    VERSION = 0
    ATTRIBUTE_FACTORY = TanV0Attribute

    def forward(self, input):
        return torch.tan(input)
