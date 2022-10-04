# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class TopKV3Attribute(Attribute):
    axis: int
    mode: str
    sort: str
    index_element_type: str = field(default="i32")

    def __post_init__(self):
        super().__post_init__()
        valid_mode = ["min", "max"]
        if self.mode not in valid_mode:
            raise ValueError(
                f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}."
            )

        valid_sort = ["value", "index", "none"]
        if self.sort not in valid_sort:
            raise ValueError(
                f"Invalid sort {self.sort}. " f"It must be one of {valid_sort}."
            )

        valid_index_element_type = ["i32", "i64"]
        if self.index_element_type not in valid_index_element_type:
            raise ValueError(
                f"Invalid index_element_type {self.index_element_type}. "
                f"It must be one of {valid_index_element_type}."
            )


@OPS.register()
class TopKV3(Operation):
    TYPE = "TopK"
    VERSION = 3
    ATTRIBUTE_FACTORY = TopKV3Attribute

    def forward(self, input, k):
        raise NotImplementedError
        values, indices = torch.topk(
            input=input,
            k=k,
            dim=self.attrs.axis,
            largest=self.attrs.mode == "max",
            sorted=True,
        )

        if self.attrs.sort == "index":
            sorted = torch.argsort(indices)
            indices = indices[sorted]
            values = values[sorted]

        return values, indices
