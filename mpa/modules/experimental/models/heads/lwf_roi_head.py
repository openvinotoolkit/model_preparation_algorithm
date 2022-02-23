import torch

from mmcv.runner import force_fp32
from mmdet.models import HEADS, build_loss
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.core.bbox.transforms import bbox2roi

from mpa.modules.utils.task_adapt import map_class_names


@HEADS.register_module()
class LwfRoIHead(StandardRoIHead):
    """LwF-enabled RoI head
    """

    def __init__(self, src_classes, dst_classes, loss_cls_lwf, loss_bbox_lwf, **kwargs):
        super().__init__(**kwargs)
        self.src_classes = src_classes
        self.dst_classes = dst_classes
        self.loss_cls_lwf = build_loss(loss_cls_lwf)
        self.loss_bbox_lwf = build_loss(loss_bbox_lwf)
        self.src2dst = torch.tensor(map_class_names(self.src_classes, self.dst_classes))
        print('LwfRoIHead init!')

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      x_t=None,
                      roi_head_t=None):
        """
        Args:
            x_t (list[Tensor]): feature maps from teacher model
            roid_head_t (StandardRoIHead): feature RoI head for LwF

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            # Student
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            # Normal BBox losses
            losses.update(bbox_results['loss_bbox'])

            # Teacher
            with torch.no_grad():
                bbox_results_t = roi_head_t._bbox_forward(x_t, bbox_results['rois'])

            # LwF losses
            lwf_losses = self.loss_lwf(
                bbox_results['labels'],
                bbox_results['cls_score'], bbox_results['bbox_pred'],
                bbox_results_t['cls_score'], bbox_results_t['bbox_pred'])
            losses.update(lwf_losses)

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_scores_t', 'bbox_preds_t'))
    def loss_lwf(self, gt_labels, cls_score, bbox_pred, cls_score_t, bbox_pred_t):
        # Classification LwF loss
        # Only for BG-labeled bboxes
        is_bg_labels = (gt_labels == len(self.dst_classes))
        cls_score = cls_score[is_bg_labels]
        cls_score_t = cls_score_t[is_bg_labels]
        cls_loss = self.loss_cls_lwf(cls_score, cls_score_t)

        # Regression LwF loss
        if not self.bbox_head.reg_class_agnostic:
            n = bbox_pred.shape[0]
            bbox_pred = bbox_pred.view(n, -1, 4)[:, self.src2dst[self.src2dst >= 0]]
            bbox_pred_t = bbox_pred_t.view(n, -1, 4)[:, self.src2dst >= 0]

        bbox_loss = self.loss_bbox_lwf(bbox_pred, bbox_pred_t)
        return dict(loss_cls_lwf=cls_loss, loss_bbox_lwf=bbox_loss)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            labels=bbox_targets[0],
            label_weights=bbox_targets[1],
            bbox_targets=bbox_targets[2],
            bbox_weights=bbox_targets[3]
        )
        return bbox_results


@HEADS.register_module()
class OffLwfRoIHead(StandardRoIHead):
    """Offline LwF-enabled RoI head
    """

    def __init__(self, src_classes, dst_classes, loss_cls_lwf, **kwargs):
        super().__init__(**kwargs)
        self.src_classes = src_classes
        self.dst_classes = dst_classes
        self.loss_cls_lwf = build_loss(loss_cls_lwf)
        self.src2dst = torch.tensor(map_class_names(self.src_classes, self.dst_classes))
        print('OffLwfRoIHead init!')

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      pseudo_labels=None):
        """
        Args:
            pseudo_labels (list[Tensor]): class proabilities for OLD classes

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    pseudo_labels,  # Only one difference from super().forward_train()
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, pseudo_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        target_labels, label_weights, target_bboxes, bbox_weights = \
            self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        # Pseudo label targets
        old_plabels = []
        old_pred_indices = []
        num_targets = 0
        for img_idx, sampling_result in enumerate(sampling_results):
            pos_plabels = pseudo_labels[img_idx][sampling_result.pos_assigned_gt_inds]
            old_indices = torch.nonzero(pos_plabels[:, -1] < 1.0, as_tuple=False)[:, -1]  # OLD class pseudo labels
            pos_plabels = pos_plabels[old_indices]
            old_plabels.append(pos_plabels)
            old_pred_indices.append(old_indices + num_targets)
            num_targets += sampling_result.bboxes.shape[0]
        old_plabels = torch.cat(old_plabels)
        old_pred_indices = torch.cat(old_pred_indices)

        label_weights[old_pred_indices] = 0.0  # Suppress x-entropy for OLD predictions

        # Ordinary NEW class loss
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        target_labels, label_weights, target_bboxes, bbox_weights)
        bbox_results.update(loss_bbox=loss_bbox)

        # LwF loss for OLD classes
        loss_cls_lwf = self.loss_cls_lwf(
                bbox_results['cls_score'][old_pred_indices], old_plabels, target_is_logit=False)
        bbox_results['loss_bbox'].update(loss_cls_lwf=loss_cls_lwf)

        return bbox_results
