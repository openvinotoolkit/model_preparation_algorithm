# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union

import torch


class BaseRecordingForwardHook(ABC):
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.
    Example::
        with BaseRecordingForwardHook(model.module.backbone) as hook:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)
    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
    """
    def __init__(self, module: torch.nn.Module, fpn_idx: int = 0) -> None:
        self._module = module
        self._handle = None
        self._records = []
        self.fpn_idx = fpn_idx

    @property
    def records(self):
        return self._records

    @abstractmethod
    def func(x: torch.Tensor, fpn_idx: int = 0) -> torch.Tensor:
        """This method get the feature vector or saliency map from the output of the module.
        Args:
            x (torch.Tensor): Feature map from the backbone module
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                    Defaults to 0 which uses the largest feature map from FPN.
        Returns:
            torch.Tensor (torch.Tensor): Saliency map for feature vector
        """
        raise NotImplementedError

    def _recording_forward(self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        tensor = self.func(output)
        tensor = tensor.detach().cpu().numpy()
        if len(tensor) > 1:
            for single_tensor in tensor:
                self._records.append(single_tensor)
        else:
            self._records.append(tensor)

    def __enter__(self) -> BaseRecordingForwardHook:
        self._handle = self._module.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()


class EigenCamHook(BaseRecordingForwardHook):
    @staticmethod
    def func(x_: torch.Tensor) -> torch.Tensor:
        x = x_.type(torch.float)
        bs, c, h, w = x.size()
        reshaped_fmap = x.reshape((bs, c, h * w)).transpose(1, 2)
        reshaped_fmap = reshaped_fmap - reshaped_fmap.mean(1)[:, None, :]
        U, S, V = torch.linalg.svd(reshaped_fmap, full_matrices=True)
        saliency_map = (reshaped_fmap @ V[:, 0][:, :, None]).squeeze(-1)
        max_values, _ = torch.max(saliency_map, -1)
        min_values, _ = torch.min(saliency_map, -1)
        saliency_map = (
            255
            * (saliency_map - min_values[:, None])
            / ((max_values - min_values + 1e-12)[:, None])
        )
        saliency_map = saliency_map.reshape((bs, h, w))
        saliency_map = saliency_map.to(torch.uint8)
        return saliency_map


class ActivationMapHook(BaseRecordingForwardHook):
    @staticmethod
    def func(feature_map: Union[torch.Tensor, list[torch.Tensor]], fpn_idx: int = 0) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255)."""
        if isinstance(feature_map, list):
            assert fpn_idx < len(feature_map), \
                f"fpn_idx: {fpn_idx} is out of scope of feature_map length {len(feature_map)}!"
            feature_map = feature_map[fpn_idx]

        bs, c, h, w = feature_map.size()
        activation_map = torch.mean(feature_map, dim=1)
        activation_map = activation_map.reshape((bs, h * w))
        max_values, _ = torch.max(activation_map, -1)
        min_values, _ = torch.min(activation_map, -1)
        activation_map = (
            255
            * (activation_map - min_values[:, None])
            / (max_values - min_values + 1e-12)[:, None]
        )
        activation_map = activation_map.reshape((bs, h, w))
        activation_map = activation_map.to(torch.uint8)
        return activation_map


class FeatureVectorHook(BaseRecordingForwardHook):
    @staticmethod
    def func(feature_map: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        """Generate the feature vector by average pooling feature maps."""
        if isinstance(feature_map, list):
            # aggregate feature maps from Feature Pyramid Network
            feature_vector = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)) for f in feature_map]
            feature_vector = torch.cat(feature_vector, 1)
        else:
            feature_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
        return feature_vector
