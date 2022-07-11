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

import torch


def get_saliency_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Generate the saliency map by average feature maps then normalizing to (0, 255)

    Args:
        feature_map (torch.Tensor): feature maps from backbone

    Returns:
        torch.Tensor: Saliency Map
    """
    saliency_map = torch.sigmoid(torch.mean(feature_map, dim=1))
    saliency_map = (
        255
        * (saliency_map - torch.min(saliency_map))
        / (torch.max(saliency_map) - torch.min(saliency_map) + 1e-12)
    )
    saliency_map = saliency_map.to(torch.uint8)
    return saliency_map


def get_feature_vector(feature_map: torch.Tensor) -> torch.Tensor:
    """Generate the feature vector by average pooling feature maps

    Args:
        feature_map (torch.Tensor): feature map from backbone

    Returns:
        torch.Tensor: feature vector(representation vector)
    """
    feature_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
    return feature_vector
