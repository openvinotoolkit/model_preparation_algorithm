import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.core.utils.misc import topk
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import Shared2FCBBoxHead
from mmdet.models.dense_heads import SSDHead

from mpa.modules.experimental.utils.pseudo_nms import pseudo_multiclass_nms


@HEADS.register_module()
class PseudoShared2FCBBoxHead(Shared2FCBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform model outputs to bboxes after NMS

        Copied from mmdet/models/roi_heads/bbox_heads/bbox_head.py
        to augment detection output with class probabilities as pseudo label
        """
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = pseudo_multiclass_nms(bboxes, scores,
                                                           cfg.score_thr, cfg.nms,
                                                           cfg.max_per_img)

            return det_bboxes, det_labels


@HEADS.register_module()
class PseudoSSDHead(SSDHead):

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.

        Copied from mmdet/models/dense_heads/anchor_head.py
        to augment detection output with class probabilities as pseudo label
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
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
                # scores = scores[:, :self.num_classes]  # Removed this line to retain BG prob
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
        det_bboxes, det_labels = pseudo_multiclass_nms(mlvl_bboxes, mlvl_scores,  # Replaced multiclass_nms by pseudo_*
                                                       cfg.score_thr, cfg.nms,
                                                       cfg.max_per_img)
        return det_bboxes, det_labels
