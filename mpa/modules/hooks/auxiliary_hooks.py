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

from typing import List, Tuple, Union

import torch
import mmcls

import mpa


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

    def _recording_forward(self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
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
    """While registered with the designated PyTorch module, this class caches saliency maps during forward pass.

    Example::
        with SaliencyMapHook(model.module.backbone) as hook:      
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)
    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
        fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                  Defaults to 0 which uses the largest feature map from FPN.
    """
    def __init__(self, module: torch.nn.Module, fpn_idx: int = 0) -> None:
        self._module = module
        self._handle = None
        self._records = []
        self._fpn_idx = fpn_idx

    @property
    def records(self):
        return self._records

    @staticmethod
    def func(feature_map: Union[torch.Tensor, List[torch.Tensor]], fpn_idx: int = 0) -> torch.Tensor:
        """Generate the saliency map by average feature maps then normalizing to (0, 255).

        Args:
            feature_map (Union[torch.Tensor, list[torch.Tensor]]): feature maps from backbone or list of feature maps 
                                                                    from FPN.
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                      Defaults to 0 which uses the largest feature map from FPN.

        Returns:
            torch.Tensor: Saliency Map
        """
        if isinstance(feature_map, list):
            feature_map = feature_map[fpn_idx]

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

    def _recording_forward(self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        saliency_map = self.func(output, self._fpn_idx)
        saliency_map = saliency_map.detach().cpu().numpy()
        for tensor in saliency_map:
            self._records.append(tensor)

    def __enter__(self) -> SaliencyMapHook:
        self._handle = self._module.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()


class SaliencyMapHookDet(SaliencyMapHook):
    """While registered with the designated PyTorch module, this class caches the saliency maps during forward pass.
        Saliency maps are generated using classification head outputs

    Example:
        with SaliencyMapHookDet(model.module) as hook:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            print(hook.records)
    """
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module.backbone)
        self._neck = module.neck if module.with_neck else None
        self._bbox_head = module.bbox_head
        self._num_cls_out_channels = module.bbox_head.cls_out_channels  # SSD-like heads also have background class
        if hasattr(module.bbox_head, 'anchor_generator'):
            self._num_anchors = module.bbox_head.anchor_generator.num_base_anchors
        else:
            self._num_anchors = [1] * 10

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
                    torch.nn.functional.interpolate(
                        cls_scores_anchorless_per_level,
                        (h, w),
                        mode='bilinear'))
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
        """Forward features through the classification head of the detector"""
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)

            if isinstance(self._bbox_head, mpa.modules.models.heads.custom_ssd_head.CustomSSDHead):
                cls_scores = []
                for feat, cls_conv in zip(x, self._bbox_head.cls_convs):
                    cls_scores.append(cls_conv(feat))
            elif isinstance(self._bbox_head, mpa.modules.models.heads.custom_atss_head.CustomATSSHead):
                cls_scores = []
                for cls_feat in x:
                    for cls_conv in self._bbox_head.cls_convs:
                        cls_feat = cls_conv(cls_feat)
                    cls_score = self._bbox_head.atss_cls(cls_feat)
                    cls_scores.append(cls_score)
            elif isinstance(self._bbox_head, mpa.modules.models.heads.custom_vfnet_head.CustomVFNetHead):
                # Not clear how to separate cls_scores from bbox_preds
                cls_scores, _, _ = self._bbox_head(x)
            elif isinstance(self._bbox_head, mpa.modules.models.heads.custom_yolox_head.CustomYOLOXHead):
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
                raise NotImplemented("Not supported detection head provided. "
                                     "SaliencyMapHookDet supports only the following single stage detectors: "
                                     "YOLOXHead, ATSSHead, SSDHead, VFNetHead.")
        return cls_scores


class ReciproCAMHook(SaliencyMapHook):
    """While registered with the designated PyTorch module, this class caches the saliency maps during forward pass.
    Saliency maps are generated using Recipro-CAM:
    Recipro-CAM: Gradient-free reciprocal class activation map, https://arxiv.org/pdf/2209.14074.pdf.

    Example::
        with ReciproCAMHook(model.module) as hook:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)
    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
        fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                  Defaults to 0 which uses the largest feature map from FPN.
    """
    def __init__(self, module: torch.nn.Module, fpn_idx: int = 0) -> None:
        super().__init__(module.backbone, fpn_idx)
        self._neck = module.neck if module.with_neck else None
        self._head = module.head
        self._num_classes = module.head.num_classes

    def func(self, feature_map: Union[torch.Tensor, List[torch.Tensor]], fpn_idx: int = 0) -> torch.Tensor:
        """Generate the saliency maps using Recipro-CAM and then normalizing to (0, 255).

        Args:
            feature_map (Union[torch.Tensor, List[torch.Tensor]]): feature maps from backbone or list of feature maps
                                                                    from FPN.
            fpn_idx (int, optional): The layer index to be processed if the model is a FPN.
                                      Defaults to 0 which uses the largest feature map from FPN.

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
        if self._neck is not None and isinstance(self._neck, mmcls.models.necks.gap.GlobalAveragePooling):
            # Optimization workaround for the GAP case (simulate GAP with more simple compute graph)
            # Possible due to static sparsity of mosaic_feature_map
            # Makes the downstream GAP operation to be dummy
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


class FeatureVectorHook:
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.

    Example::
        with FeatureVectorHook(model.module.backbone) as hook:      
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)
    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
    """
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self._handle = None
        self._records = []

    @property
    def records(self):
        return self._records

    @staticmethod
    def func(feature_map: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        """Generate the feature vector by average pooling feature maps.

        If the input is a list of feature maps from FPN, per-layer feature vector is first generated by averaging 
        feature maps in each FPN layer, then concatenate all the per-layer feature vector as the final result.

        Args:
            feature_map (Union[torch.Tensor, list[torch.Tensor]]): feature maps from backbone or list of feature maps 
                                                                    from FPN.

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

    def _recording_forward(self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
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
