import torch

from mmdet.core import multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolox_head import YOLOXHead


@HEADS.register_module()
class CustomYOLOXHead(YOLOXHead):
    def __init__(
        self,
        calib_scale=0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.calib_scale = calib_scale

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """
        Add number of instances per category calibration to mmdet yolox_head._get_bboxes_single
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, cls_scores[0].device, with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        # original_score = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_cls_scores = (torch.cat(flatten_cls_scores, dim=1) - self.calib_scale).sigmoid()

        # # For debug
        # pos_inds = original_score[0].max(dim=1)[0].topk(k=5)[1]
        # print(f'\n{original_score[0][pos_inds]}\n==>\n{flatten_cls_scores[0][pos_inds]}\n')

        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(scale_factors)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]

            result_list.append(
                multiclass_nms(bboxes, cls_scores, cfg.score_thr, cfg.nms,
                               max_num=cfg.max_per_img, score_factors=score_factor))

        return result_list
