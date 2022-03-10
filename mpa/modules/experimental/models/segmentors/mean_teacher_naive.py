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
    def __init__(self, ori_type=None, unsup_weight=0.1, **kwargs):
        print('MeanTeacherNaive Segmentor init!')
        super(MeanTeacherNaive, self).__init__()

        cfg = kwargs.copy()
        if ori_type == 'SemiSLSegmentor':
            cfg['type'] = 'SemiSLSegmentor'
            self.align_corners = cfg['decode_head'][-1].align_corners
        else:
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
        return self.model_t.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model_t.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model_t.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        return self.model_t.forward_dummy(img, **kwargs)

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        ul_img = kwargs['ul_img']
        ul_img_metas = kwargs['ul_img_metas']

        with torch.no_grad():
            teacher_feat = self.model_t.extract_feat(ul_img)
            teacher_logit = self.model_t._decode_head_forward_test(teacher_feat, ul_img_metas)
            teacher_logit = resize(input=teacher_logit,
                                   size=ul_img.shape[2:],
                                   mode='bilinear',
                                   align_corners=self.align_corners)
            conf_from_teacher, pl_from_teacher = torch.max(torch.softmax(teacher_logit, axis=1), axis=1, keepdim=True)

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u = self.model_s.extract_feat(ul_img)
        loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        loss_decode_u = self.model_s._decode_head_forward_train(x_u, ul_img_metas, pl_from_teacher)

        for key in loss_decode_u.keys():
            losses[key] = (loss_decode[key] + loss_decode_u[key]*self.unsup_weight)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect teacher model as output state_dict (student as auxilliary)
        """
        logger.info('----------------- MeanTeacherNaive.state_dict_hook() called')
        output = OrderedDict()
        for k, v in state_dict.items():
            if 'model_t.' in k:
                k = k.replace('model_t.', '')
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
