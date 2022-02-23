import torch

from mmcv.runner import force_fp32
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import RPNHead
from mmdet.core import multi_apply


@HEADS.register_module()
class LwfRPNHead(RPNHead):
    """LwF-enabled RPN head
    """

    def __init__(self, in_channels, loss_cls_lwf, loss_bbox_lwf, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.loss_cls_lwf = build_loss(loss_cls_lwf)
        self.loss_bbox_lwf = build_loss(loss_bbox_lwf)
        print('LwfRPNHead init!')

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      teacher_rpn_outputs=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):

        # Normal losses & proposals
        outputs = self(x)
        if gt_labels is None:
            loss_inputs = outputs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outputs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # Compute LwF losses
        lwf_losses = self.loss_lwf(*outputs, *teacher_rpn_outputs)
        losses.update(lwf_losses)

        # Proposals
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outputs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_scores_t', 'bbox_preds_t'))
    def loss_lwf(self, cls_scores, bbox_preds, cls_scores_t, bbox_preds_t):
        """Compute multi-scale LwF losses

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            cls_scores_t (list[Tensor]): Box scores of teacher model
            bbox_preds_t (list[Tensor]): Box energies / deltas of teacher model

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        cls_losses, bbox_losses = multi_apply(
            self.loss_lwf_single,
            cls_scores,
            bbox_preds,
            cls_scores_t,
            bbox_preds_t
        )
        return dict(loss_rpn_cls_lwf=cls_losses, loss_rpn_bbox_lwf=bbox_losses)

    def loss_lwf_single(self, cls_score, bbox_pred, cls_score_t, bbox_pred_t):
        """Compute single-scale LwF losses
        """
        # Classification LwF loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        cls_score_t = cls_score_t.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        cls_prob_t = torch.sigmoid(cls_score_t)
        cls_loss = self.loss_cls_lwf(cls_score, cls_score_t,
                                     weight=cls_prob_t)  # More weights on teacher's positives

        # Regression LwF loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_t = bbox_pred_t.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_loss = self.loss_bbox_lwf(bbox_pred, bbox_pred_t,
                                       weight=cls_prob_t)  # More weights on teacher's positives

        return cls_loss, bbox_loss
