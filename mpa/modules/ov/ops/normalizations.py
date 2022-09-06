# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

from torch.nn import functional as F

from .builder import OPS
from .op import Attribute, Operation
from .poolings import AvgPoolV1


@dataclass
class BatchNormalizationV0Attribute(Attribute):
    epsilon: float
    max_init_iter: int = field(default=10)


@OPS.register()
class BatchNormalizationV0(Operation):
    TYPE = "BatchNormInference"
    VERSION = 0
    ATTRIBUTE_FACTORY = BatchNormalizationV0Attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._initialized = False
        self._num_init_iter = 0

    def forward(self, input, gamma, beta, mean, variance):
        output = F.batch_norm(
            input=input,
            running_mean=mean,
            running_var=variance,
            weight=gamma,
            bias=beta,
            training=self.training,
            eps=self.attrs.epsilon,
        )

        if self.training and not self._initialized:
            n_dims = input.dim() - 2
            gamma = gamma.unsqueeze(0)
            beta = beta.unsqueeze(0)
            for _ in range(n_dims):
                gamma = gamma.unsqueeze(-1)
                beta = beta.unsqueeze(-1)
            output = input * gamma + beta
            self._num_init_iter += 1
            if self._num_init_iter >= self.attrs.max_init_iter:
                gamma.data = gamma.data * mean
                beta.data = beta.data + (mean / (variance + self.attrs.epsilon))
                self._initialized = True

        return output


@dataclass
class LocalResponseNormalizationV0Attribute(Attribute):
    alpha: float
    beta: float
    bias: float
    size: int


@OPS.register()
class LocalResponseNormalizationV0(Operation):
    TYPE = "LRN"
    VERSION = 0
    ATTRIBUTE_FACTORY = LocalResponseNormalizationV0Attribute

    def forward(self, input, axes):
        dim = input.dim()

        axes = axes.detach().cpu().tolist()
        assert all(ax >= 1 for ax in axes)

        axes = [ax - 1 for ax in axes]
        kernel = [1 for _ in range(dim - 1)]
        stride = [1 for _ in range(dim - 1)]
        pads_begin = [0 for _ in range(dim - 1)]
        pads_end = [0 for _ in range(dim - 1)]
        for ax in axes:
            kernel[ax] = self.attrs.size
            pads_begin[ax] = self.attrs.size // 2
            pads_end[ax] = (self.attrs.size - 1) // 2

        avg_attrs = {
            "auto_pad": "explicit",
            "strides": stride,
            "kernel": kernel,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "exclude-pad": True,
            "shape": self.shape,
        }
        avg_pool = AvgPoolV1("temp", **avg_attrs)

        div = input.mul(input).unsqueeze(1)
        div = avg_pool(div)
        div = div.squeeze(1)
        div = div.mul(self.attrs.alpha).add(self.attrs.bias).pow(self.attrs.beta)
        output = input / div
        return output


@dataclass
class NormalizeL2V0Attribute(Attribute):
    eps: float
    eps_mode: str

    def __post_init__(self):
        super().__post_init__()
        valid_eps_mode = ["add", "max"]
        if self.eps_mode not in valid_eps_mode:
            raise ValueError(
                f"Invalid eps_mode {self.eps_mode}. "
                f"It must be one of {valid_eps_mode}."
            )


@OPS.register()
class NormalizeL2V0(Operation):
    TYPE = "NormalizeL2"
    VERSION = 0
    ATTRIBUTE_FACTORY = NormalizeL2V0Attribute

    def forward(self, input, axes):
        raise NotImplementedError
