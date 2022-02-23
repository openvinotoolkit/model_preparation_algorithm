import torch

from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor


@SEGMENTORS.register_module()
class OverSamplingWrapper(BaseSegmentor):
    def __init__(self, orig_type, **kwargs):
        super(OverSamplingWrapper, self).__init__()

        cfg = kwargs.copy()
        cfg['type'] = orig_type
        self.segmentor = build_segmentor(cfg)

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        img = torch.cat((img, kwargs.pop('os_img')), dim=0)
        img_metas += kwargs.pop('os_img_metas')
        gt_semantic_seg = torch.cat((gt_semantic_seg, kwargs.pop('os_gt_semantic_seg')), dim=0)

        segmentor = self.segmentor

        x = segmentor.extract_feat(img)
        losses = segmentor._decode_head_forward_train(x, img_metas, gt_semantic_seg)

        return losses

    def simple_test(self, img, img_meta, **kwargs):
        return self.segmentor.simple_test(img, img_meta, **kwargs)
