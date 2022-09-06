# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import List

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class InterpolateV4Attribute(Attribute):
    mode: str
    shape_calculation_mode: str
    coordinate_transformation_mode: str = field(default="half_pixel")
    nearest_mode: str = field(default="round_prefer_floor")
    antialias: bool = field(default=False)
    pads_begin: List[int] = field(default_factory=lambda: [0])
    pads_end: List[int] = field(default_factory=lambda: [0])
    cube_coeff: float = field(default=-0.75)

    def __post_init__(self):
        super().__post_init__()
        valid_mode = ["nearest", "linear", "linear_onnx", "cubic"]
        if self.mode not in valid_mode:
            raise ValueError(
                f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}."
            )
        valid_shape_calculation_mode = ["sizes", "scales"]
        if self.shape_calculation_mode not in valid_shape_calculation_mode:
            raise ValueError(
                f"Invalid shape_calculation_mode {self.shape_calculation_mode}. "
                f"It must be one of {valid_shape_calculation_mode}."
            )
        valid_coordinate_transformation_mode = [
            "half_pixel",
            "pytorch_half_pixel",
            "asymmetric",
            "tf_half_pixel_for_nn",
            "align_corners",
        ]
        if (
            self.coordinate_transformation_mode
            not in valid_coordinate_transformation_mode
        ):
            raise ValueError(
                f"Invalid coordinate_transformation_mode {self.coordinate_transformation_mode}. "
                f"It must be one of {valid_coordinate_transformation_mode}."
            )
        valid_nearest_mode = [
            "round_prefer_flow",
            "round_prefer_ceil",
            "floor",
            "ceil",
            "simple",
        ]
        if self.nearest_mode not in valid_nearest_mode:
            raise ValueError(
                f"Invalid nearest_mode {self.nearest_mode}. "
                f"It must be one of {valid_nearest_mode}."
            )


@OPS.register()
class InterpolateV4(Operation):
    TYPE = "Interpolate"
    VERSION = 4
    ATTRIBUTE_FACTORY = InterpolateV4Attribute

    def forward(self, input, sizes, scales, axes=None):
        raise NotImplementedError
