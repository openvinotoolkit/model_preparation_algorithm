import torch
# import torch.nn.functional as F

from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
import numpy as np
from .cutmix.gt_utils import generate_cutmix_mask
import functools
from collections import OrderedDict
from mpa.utils import logger


@SEGMENTORS.register_module()
class CutmixSegGTNaive(BaseSegmentor):
    def __init__(self, ori_type=None, unsup_weight=0.1, **kwargs):
        print('CutmixSegGTNaive init!')
        super(CutmixSegGTNaive, self).__init__()

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
        # gt config
        self.area_thresh = 0.0001
        self.area_thresh2 = 0.0001
        self.no_pad = False
        self.no_slim = False
        self.num_classes = kwargs['decode_head'][0]['num_classes']
        self.class_criterion_seg = np.zeros([3, self.num_classes]).astype(float)
        
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
        if 'to_be_inserted_img' in kwargs.keys():
            img_to_be_inserted = kwargs['to_be_inserted_img']
            l_gt = kwargs['to_be_inserted_gt_semantic_seg']
        else:
            img_to_be_inserted = img
            l_gt = gt_semantic_seg

        if 'sample_cat' in kwargs.keys():
            sample_cat = kwargs['sample_cat']
        else:
            sample_cat = []
            for i in range(img.shape[0]):
                each_sample_cat = np.random.randint(1, self.num_classes+1)
                sample_cat.append(each_sample_cat)
            sample_cat = torch.Tensor(sample_cat)

        with torch.no_grad():
            teacher_feat = self.model_t.extract_feat(ul_img)
            teacher_logit = self.model_t._decode_head_forward_test(teacher_feat, ul_img_metas)
            teacher_logit = resize(input=teacher_logit,
                                   size=ul_img.shape[2:],
                                   mode='bilinear',
                                   align_corners=self.align_corners)
            conf_from_teacher, pl_from_teacher = torch.max(torch.softmax(teacher_logit, axis=1), axis=1, keepdim=True)

        masks = generate_cutmix_mask(l_gt.cpu().numpy(), sample_cat.cpu().numpy(),
                                     self.area_thresh,
                                     no_pad=self.no_pad, no_slim=self.no_slim)
        if ul_img.is_cuda:
            masks = masks.cuda()
        ul_img_cutmix = ul_img * (1 - masks) + img_to_be_inserted * masks
        # labels for cutmix
        tmp = masks * l_gt
        pl_from_teacher_cutmixed = (1 - masks) * pl_from_teacher + tmp
        pl_from_teacher_cutmixed = pl_from_teacher_cutmixed.long()

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u_cutmixed = self.model_s.extract_feat(ul_img_cutmix)
        loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        loss_decode_u = self.model_s._decode_head_forward_train(x_u_cutmixed, ul_img_metas, pl_from_teacher_cutmixed)

        for key in loss_decode_u.keys():
            losses[key] = (loss_decode[key] + loss_decode_u[key]*self.unsup_weight)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect teacher model as output state_dict (student as auxilliary)
        """
        logger.info('----------------- CutmixSegGTNaive.state_dict_hook() called')
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
        logger.info('----------------- CutmixSegGTNaive.load_state_dict_pre_hook() called')
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if 'model_s.' not in k:
                state_dict['model_s.'+k] = v
                state_dict['model_t.'+k] = v
