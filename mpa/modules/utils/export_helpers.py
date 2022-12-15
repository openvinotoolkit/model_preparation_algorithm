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

from typing import List
import torch


def get_saliency_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Generate the saliency map by normalizing the feature map to (0, 255)

    Args:
        feature_map (torch.Tensor): feature map from backbone

    Returns:
        torch.Tensor: Saliency Map
    """
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


def get_feature_vector(feature_maps: List[torch.Tensor]) -> torch.Tensor:
    """Generate the feature vector by average pooling feature maps

    Args:
        feature_maps (torch.Tensor): list of feature maps from backbone

    Returns:
        torch.Tensor: feature vector(representation vector)
    """
    pooled_features = [
        torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
        for feat_map in feature_maps
    ]
    feature_vector = torch.cat(pooled_features, dim=1)
    return feature_vector
