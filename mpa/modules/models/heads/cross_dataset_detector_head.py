# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.core import images_to_levels, multi_apply

from mpa.modules.models.losses.cross_focal_loss import CrossSigmoidFocalLoss


@HEADS.register_module()
class CrossDatasetDetectorHead(BaseDenseHead):
    def get_atss_targets(self,
                         anchor_list,
                         valid_flag_list,
                         gt_bboxes_list,
                         img_metas,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         unmap_outputs=True):
        """Get targets for Detection head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        However, if the detector's head loss uses CrossSigmoidFocalLoss,
        the labels_weights_list consists of (binarized label schema * weights) of batch images
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            img_gt_set = self._get_gt_labels_binary_set(gt_labels_list, all_labels, all_label_weights)
            label_weights_list = images_to_levels(img_gt_set, num_level_anchors)
        else:
            label_weights_list = images_to_levels(all_label_weights,
                                                  num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        label_weights, bbox_weights = None, None
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            label_weights_list = self._get_gt_labels_binary_set(gt_labels_list, labels_list)
            label_weights_list = [
                label_weights.reshape(-1) for label_weights in label_weights_list
            ]
            label_weights = torch.cat(label_weights_list)
        return concat_lvl_labels, label_weights, concat_lvl_bbox_targets, bbox_weights

    def _get_gt_labels_binary_set(self, gt_labels_list, all_labels, all_label_weights=None):
        x = [0 for _ in gt_labels_list]
        for i, gt_labels in enumerate(gt_labels_list):
            for gt in set(gt_labels.tolist()):
                x[i] += float(2**gt)
            x[i] = torch.tensor([x[i]] * len(all_labels[i]), dtype=torch.float32)
            if torch.cuda.is_available():
                x[i] = x[i].cuda()
            if all_label_weights is not None:
                x[i] *= all_label_weights[i].float()
        return x

    def vfnet_to_atss_targets(self,
                              cls_scores,
                              mlvl_points,
                              gt_bboxes,
                              gt_labels,
                              img_metas,
                              gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_atss_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights
