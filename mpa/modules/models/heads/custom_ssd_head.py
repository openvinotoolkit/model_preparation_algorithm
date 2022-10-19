# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch

from mmdet.core import multiclass_nms
from mmdet.core.utils.misc import topk
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import smooth_l1_loss
from mmdet.models.dense_heads.ssd_head import SSDHead


@HEADS.register_module()
class CustomSSDHead(SSDHead):
    def __init__(
        self,
        *args,
        bg_loss_weight=-1.0,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0
        ),
        calib_scale=0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_cls = build_loss(loss_cls)
        self.bg_loss_weight = bg_loss_weight
        self.calib_scale = calib_scale

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.num_classes)).nonzero().reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero().view(-1)

        # Re-weigting BG loss
        label_weights = label_weights.reshape(-1)
        if self.bg_loss_weight >= 0.0:
            neg_indices = (labels == self.num_classes)
            label_weights = label_weights.clone()
            label_weights[neg_indices] = self.bg_loss_weight

        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)
        if len(loss_cls_all.shape) > 1:
            loss_cls_all = loss_cls_all.sum(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """
        Add number of instances per category calibration to mmdet anchor_head._get_bboxes_single
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = (cls_score - self.calib_scale).sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                # original_score = cls_score.softmax(-1)[:, :self.num_classes]
                cls_score[:, :self.num_classes] = cls_score[:, :self.num_classes] - self.calib_scale
                scores = cls_score.softmax(-1)
                scores = scores[:, :self.num_classes]

                # # For debug
                # pos_inds = original_score.max(dim=1)[0].topk(5)[1]
                # print(f'\n{original_score[pos_inds]} \n==>\n {scores[pos_inds]}\n')

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = topk(max_scores, nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)

        nms_pre_classwise = cfg.get('nms_pre_classwise', -1)
        if nms_pre_classwise > 0:
            if cfg.get('nms_pre', -1) > 0:
                raise RuntimeError('nms_pre and nms_pre_classwise are mutually exclusive.')
            new_mlvl_scores = []
            new_mlvl_boxes = []
            for class_id in range(self.num_classes):
                _, topk_inds = topk(mlvl_scores[:, class_id], nms_pre_classwise)
                boxes = mlvl_bboxes[topk_inds]
                top_scores = mlvl_scores[topk_inds]
                scores = torch.zeros_like(top_scores)
                scores[:, class_id] = top_scores[:, class_id]

                new_mlvl_scores.append(scores)
                new_mlvl_boxes.append(boxes)

            mlvl_scores = torch.cat(new_mlvl_scores)
            mlvl_bboxes = torch.cat(new_mlvl_boxes)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores
