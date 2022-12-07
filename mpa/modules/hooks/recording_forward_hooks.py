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
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F

from mmcls.models.necks.gap import GlobalAveragePooling
from mpa.modules.models.heads.custom_atss_head import CustomATSSHead
from mpa.modules.models.heads.custom_ssd_head import CustomSSDHead
from mpa.modules.models.heads.custom_vfnet_head import CustomVFNetHead
from mpa.modules.models.heads.custom_yolox_head import CustomYOLOXHead


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
        self._fpn_idx = fpn_idx

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
        self._handle = self._module.backbone.register_forward_hook(self._recording_forward)
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
            feature_vector = [F.adaptive_avg_pool2d(f, (1, 1)) for f in feature_map]
            feature_vector = torch.cat(feature_vector, 1)
        else:
            feature_vector = F.adaptive_avg_pool2d(feature_map, (1, 1))
        return feature_vector


class DetSaliencyMapHook(BaseRecordingForwardHook):
    """Saliency map hook for object detection models."""
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module)
        self._neck = module.neck if module.with_neck else None
        self._bbox_head = module.bbox_head
        self._num_cls_out_channels = module.bbox_head.cls_out_channels  # SSD-like heads also have background class
        if hasattr(module.bbox_head, 'anchor_generator'):
            self._num_anchors = module.bbox_head.anchor_generator.num_base_anchors
        else:
            self._num_anchors = [1] * 10

    @staticmethod
    def func(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]], _: int = 0,
             cls_scores_provided: bool = False) -> torch.Tensor:
        """
        Generate the saliency map from raw classification head output, then normalizing to (0, 255).

        :param x: Feature maps from backbone/FPN or classification scores from cls_head
        :param cls_scores_provided: If True - use 'x' as is, otherwise forward 'x' through the classification head
        :return: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        if cls_scores_provided:
            cls_scores = x
        else:
            cls_scores = self._get_cls_scores_from_feature_map(x)

        bs, _, h, w = cls_scores[-1].size()
        saliency_maps = torch.empty(bs, self._num_cls_out_channels, h, w)
        for batch_idx in range(bs):
            cls_scores_anchorless = []
            for scale_idx, cls_scores_per_scale in enumerate(cls_scores):
                cls_scores_anchor_grouped = cls_scores_per_scale[batch_idx].reshape(
                    self._num_anchors[scale_idx],
                    (self._num_cls_out_channels),
                    *cls_scores_per_scale.shape[-2:]
                )
                cls_scores_out, _ = cls_scores_anchor_grouped.max(dim=0)
                cls_scores_anchorless.append(cls_scores_out.unsqueeze(0))
            cls_scores_anchorless_resized = []
            for cls_scores_anchorless_per_level in cls_scores_anchorless:
                cls_scores_anchorless_resized.append(
                    F.interpolate(
                        cls_scores_anchorless_per_level,
                        (h, w),
                        mode='bilinear'
                    )
                )
            saliency_maps[batch_idx] = torch.cat(cls_scores_anchorless_resized, dim=0).mean(dim=0)

        saliency_maps = saliency_maps.reshape((bs, self._num_cls_out_channels, -1))
        max_values, _ = torch.max(saliency_maps, -1)
        min_values, _ = torch.min(saliency_maps, -1)
        saliency_maps = (
                255
                * (saliency_maps - min_values[:, :, None])
                / (max_values - min_values + 1e-12)[:, :, None]
        )
        saliency_maps = saliency_maps.reshape((bs, self._num_cls_out_channels, h, w))
        saliency_maps = saliency_maps.to(torch.uint8)
        return saliency_maps

    def _get_cls_scores_from_feature_map(self, x: torch.Tensor) -> List:
        """Forward features through the classification head of the detector."""
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)

            if isinstance(self._bbox_head, CustomSSDHead):
                cls_scores = []
                for feat, cls_conv in zip(x, self._bbox_head.cls_convs):
                    cls_scores.append(cls_conv(feat))
            elif isinstance(self._bbox_head, CustomATSSHead):
                cls_scores = []
                for cls_feat in x:
                    for cls_conv in self._bbox_head.cls_convs:
                        cls_feat = cls_conv(cls_feat)
                    cls_score = self._bbox_head.atss_cls(cls_feat)
                    cls_scores.append(cls_score)
            elif isinstance(self._bbox_head, CustomVFNetHead):
                # Not clear how to separate cls_scores from bbox_preds
                cls_scores, _, _ = self._bbox_head(x)
            elif isinstance(self._bbox_head, CustomYOLOXHead):
                def forward_single(x, cls_convs, conv_cls):
                    """Forward feature of a single scale level."""
                    cls_feat = cls_convs(x)
                    cls_score = conv_cls(cls_feat)
                    return cls_score

                map_results = map(forward_single, x,
                                  self._bbox_head.multi_level_cls_convs,
                                  self._bbox_head.multi_level_conv_cls)
                cls_scores = list(map_results)
            else:
                raise NotImplementedError("Not supported detection head provided. "
                                          "DetSaliencyMapHook supports only the following single stage detectors: "
                                          "YOLOXHead, ATSSHead, SSDHead, VFNetHead.")
        return cls_scores


class ReciproCAMHook(BaseRecordingForwardHook):
    """
    Implementation of recipro-cam for class-wise saliency map
    recipro-cam: gradient-free reciprocal class activation map (https://arxiv.org/pdf/2209.14074.pdf)
    """
    def __init__(self, module: torch.nn.Module, fpn_idx: int = 0) -> None:
        super().__init__(module.backbone, fpn_idx)
        self._neck = module.neck if module.with_neck else None
        self._head = module.head
        self._num_classes = module.head.num_classes

    @staticmethod
    def func(self, feature_map: Union[torch.Tensor, List[torch.Tensor]], fpn_idx: int = 0) -> torch.Tensor:
        """
        Generate the class-wise saliency maps using Recipro-CAM and then normalizing to (0, 255).

        Returns:
            torch.Tensor: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        if isinstance(feature_map, list):
            feature_map = feature_map[fpn_idx]

        bs, c, h, w = feature_map.size()

        saliency_maps = torch.empty(bs, self._num_classes, h, w)
        for f in range(bs):
            mosaic_feature_map = self._get_mosaic_feature_map(feature_map[f], c, h, w)
            mosaic_prediction = self._predict_from_feature_map(mosaic_feature_map)
            saliency_maps[f] = mosaic_prediction.transpose(0, 1).reshape((self._num_classes, h, w))

        saliency_maps = saliency_maps.reshape((bs, self._num_classes, h * w))
        max_values, _ = torch.max(saliency_maps, -1)
        min_values, _ = torch.min(saliency_maps, -1)
        saliency_maps = (
            255
            * (saliency_maps - min_values[:, :, None])
            / (max_values - min_values + 1e-12)[:, :, None]
        )
        saliency_maps = saliency_maps.reshape((bs, self._num_classes, h, w))
        saliency_maps = saliency_maps.to(torch.uint8)
        return saliency_maps

    def _predict_from_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)
            logits = self._head.simple_test(x)
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits)
        return logits

    def _get_mosaic_feature_map(self, feature_map: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
        if self._neck is not None and isinstance(self._neck, GlobalAveragePooling):
            """
            Optimization workaround for the GAP case (simulate GAP with more simple compute graph)
            Possible due to static sparsity of mosaic_feature_map
            Makes the downstream GAP operation to be dummy
            """
            feature_map_transposed = torch.flatten(feature_map, start_dim=1).transpose(0, 1)[:, :, None, None]
            mosaic_feature_map = feature_map_transposed / (h * w)
        else:
            feature_map_repeated = feature_map.repeat(h * w, 1, 1, 1)
            mosaic_feature_map_mask = torch.zeros(h * w, c, h, w).to(feature_map.device)
            spacial_order = torch.arange(h * w).reshape(h, w)
            for i in range(h):
                for j in range(w):
                    k = spacial_order[i, j]
                    mosaic_feature_map_mask[k, :, i, j] = torch.ones(c).to(feature_map.device)
            mosaic_feature_map = feature_map_repeated * mosaic_feature_map_mask
        return mosaic_feature_map
