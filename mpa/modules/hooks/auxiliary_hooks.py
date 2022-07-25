# Copyright (C) 2021 Intel Corporation
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
from typing import Union

import torch


class EigenCamHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self._handle = None
        self._records = []

    @property
    def records(self):
        return self._records

    @staticmethod
    def func(x: torch.Tensor) -> torch.Tensor:
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

    def _recording_forward(
        self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:

        saliency_map = self.func(output)
        saliency_map = saliency_map.detach().cpu().numpy()
        if len(saliency_map) > 1:
            for tensor in saliency_map:
                self._records.append(tensor)
        else:
            self._records.append(saliency_map)

    def __enter__(self) -> SaliencyMapHook:
        self._handle = self._module.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()


class SaliencyMapHook:
    # TODO[EUGENE]: improve docstring for this class
    """_summary_

    Args:
        module (torch.nn.Module): _description_
        _fpn_idx (int, optional): _description_. Defaults to 0.
    """
    def __init__(self, module: torch.nn.Module, _fpn_idx: int = 0) -> None:
        self._module = module
        self._handle = None
        self._records = []
        self._fpn_idx = _fpn_idx

    @property
    def records(self):
        return self._records

    @staticmethod
    def func(feature_map: Union[torch.Tensor, list[torch.Tensor]], _fpn_idx: int = 0) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255)

        Args:
            feature_map (Union[torch.Tensor, list[torch.Tensor]]): feature maps from backbone
            _fpn_idx (int): 

        Returns:
            torch.Tensor: Saliency Map
        """
        if isinstance(feature_map, list):
            feature_map = feature_map[_fpn_idx]

        bs, c, h, w = feature_map.size()
        saliency_map = torch.mean(feature_map, dim=1)
        saliency_map = saliency_map.reshape((bs, h * w))
        max_values, _ = torch.max(saliency_map, -1)
        min_values, _ = torch.min(saliency_map, -1)
        saliency_map = (
            255
            * (saliency_map - min_values[:, None])
            / (max_values - min_values + 1e-12)[:, None]
        )
        saliency_map = saliency_map.reshape((bs, h, w))
        saliency_map = saliency_map.to(torch.uint8)
        return saliency_map

    def _recording_forward(
        self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        saliency_map = self.func(output, self._fpn_idx)
        saliency_map = saliency_map.detach().cpu().numpy()
        for tensor in saliency_map:
            self._records.append(tensor)

    def __enter__(self) -> SaliencyMapHook:
        self._handle = self._module.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()


class FeatureVectorHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self._handle = None
        self._records = []

    @property
    def records(self):
        return self._records

    # TODO[EUGENE]: improve docstring
    @staticmethod
    def func(feature_map: torch.Tensor) -> torch.Tensor:
        """Generate the feature vector by average pooling feature maps

        Args:
            feature_map (torch.Tensor): feature map from backbone

        Returns:
            torch.Tensor: feature vector(representation vector)
        """ 
        if isinstance(feature_map, list):
            # aggregate feature maps from Feature Pyramid Network
            feature_vector = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)) for f in feature_map]
            feature_vector = torch.cat(feature_vector, 1)
        else:
            feature_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
        return feature_vector        

    def _recording_forward(
        self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        feature_vector = self.func(output)
        feature_vector = feature_vector.detach().cpu().numpy()
        if len(feature_vector) > 1:
            for tensor in feature_vector:
                self._records.append(tensor)
        else:
            self._records.append(feature_vector)

    def __enter__(self) -> FeatureVectorHook:
        self._handle = self._module.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
