import torch

from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
from .cutmix.mask_gen import BoxMaskGenerator
import functools
from collections import OrderedDict
from mpa.utils import logger


@SEGMENTORS.register_module()
class CutmixSegNaive(BaseSegmentor):
    def __init__(self, ori_type=None, unsup_weight=0.1, **kwargs):
        print('CutmixSegNaive init!')
        super(CutmixSegNaive, self).__init__()

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
        self.mask_generator = BoxMaskGenerator()
        
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
        ul1_img = kwargs['ul_img']
        ul1_img_metas = kwargs['ul_img_metas']
        ul2_img = kwargs['ul2_img']
        ul2_img_metas = kwargs['ul2_img_metas']

        mask_size = ul1_img.shape[2:]
        n_masks = ul1_img.shape[0]
        masks = torch.Tensor(self.mask_generator.generate_params(n_masks, mask_size))
        if ul1_img.is_cuda:
            masks = masks.cuda()
        ul_img_cutmix = (1-masks) * ul1_img + masks * ul2_img

        with torch.no_grad():
            ul1_feat = self.model_t.extract_feat(ul1_img)
            ul1_logit = self.model_t._decode_head_forward_test(ul1_feat, ul1_img_metas)
            ul1_logit = resize(input=ul1_logit,
                               size=ul1_img.shape[2:],
                               mode='bilinear',
                               align_corners=self.align_corners)
            ul1_conf, ul1_pl = torch.max(torch.softmax(ul1_logit, axis=1), axis=1, keepdim=True)

            ul2_feat = self.model_t.extract_feat(ul2_img)
            ul2_logit = self.model_t._decode_head_forward_test(ul2_feat, ul2_img_metas)
            ul2_logit = resize(input=ul2_logit,
                               size=ul2_img.shape[2:],
                               mode='bilinear',
                               align_corners=self.align_corners)
            ul2_conf, ul2_pl = torch.max(torch.softmax(ul2_logit, axis=1), axis=1, keepdim=True)

            pl_cutmixed = (1-masks)*ul1_pl + masks*ul2_pl
            pl_cutmixed = pl_cutmixed.long()

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u_cutmixed = self.model_s.extract_feat(ul_img_cutmix)
        loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        loss_decode_u = self.model_s._decode_head_forward_train(x_u_cutmixed, ul1_img_metas, pl_cutmixed)

        for key in loss_decode_u.keys():
            losses[key] = (loss_decode[key] + loss_decode_u[key]*self.unsup_weight)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect teacher model as output state_dict (student as auxilliary)
        """
        logger.info('----------------- CutmixSegNaive.state_dict_hook() called')
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
        logger.info('----------------- CutmixSegNaive.load_state_dict_pre_hook() called')
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if 'model_s.' not in k:
                state_dict['model_s.'+k] = v
                state_dict['model_t.'+k] = v
