from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
import torch
import functools
from collections import OrderedDict
from mpa.utils.logger import get_logger
logger = get_logger()


@SEGMENTORS.register_module()
class MeanTeacherNaive(BaseSegmentor):
    def __init__(self, ori_type=None, unsup_weight=0.1, warmup_start_iter=30, **kwargs):
        logger.info('MeanTeacherNaive Segmentor init!')
        super(MeanTeacherNaive, self).__init__()
        self.test_cfg = kwargs['test_cfg']
        self.warmup_start_iter = warmup_start_iter
        self.count_iter = 0

        cfg = kwargs.copy()
        if ori_type == 'EncoderDecoder':
            cfg['type'] = 'EncoderDecoder'
            self.align_corners = cfg['decode_head'].align_corners
        self.model_s = build_segmentor(cfg)
        self.model_t = build_segmentor(cfg)
        self.unsup_weight = unsup_weight

        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(
            functools.partial(self.load_state_dict_pre_hook, self)
        )
    
    def extract_feat(self, imgs):
        return self.model_s.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model_s.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model_s.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        return self.model_s.forward_dummy(img, **kwargs)

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        self.count_iter += 1
        if self.warmup_start_iter > self.count_iter:
            x = self.model_s.extract_feat(img)
            loss_decode, _ = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
            return loss_decode

        ul_data = kwargs['extra_0']
        ul_s_img = ul_data['img']
        ul_w_img = ul_data['ul_w_img']
        ul_img_metas = ul_data['img_metas']

        with torch.no_grad():
            teacher_feat = self.model_t.extract_feat(ul_w_img)
            teacher_logit = self.model_t._decode_head_forward_test(teacher_feat, ul_img_metas)
            teacher_logit = resize(input=teacher_logit,
                                   size=ul_w_img.shape[2:],
                                   mode='bilinear',
                                   align_corners=self.align_corners)
            conf_from_teacher, pl_from_teacher = torch.max(torch.softmax(teacher_logit, axis=1), axis=1, keepdim=True)        

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u = self.model_s.extract_feat(ul_s_img)
        loss_decode, _ = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
        loss_decode_u, _ = self.model_s._decode_head_forward_train(x_u, ul_img_metas, gt_semantic_seg=pl_from_teacher)

        for (k, v) in loss_decode_u.items():
            if v is None:
                continue
            losses[k] = (loss_decode[k] + loss_decode_u[k]*self.unsup_weight)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect student model as output state_dict (teacher as auxilliary)
        """
        logger.info('----------------- MeanTeacherNaive.state_dict_hook() called')
        output = OrderedDict()
        for k, v in state_dict.items():
            if 'model_s.' in k:
                k = k.replace('model_s.', '')
                output[k] = v
        return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to teacher model
        """
        logger.info('----------------- MeanTeacherNaive.load_state_dict_pre_hook() called')
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if 'model_s.' not in k:
                state_dict['model_s.'+k] = v
                state_dict['model_t.'+k] = v
