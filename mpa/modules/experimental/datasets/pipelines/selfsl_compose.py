from copy import deepcopy
import numpy as np

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.pipelines import to_tensor
from mmcv.utils import build_from_cfg


@PIPELINES.register_module()
class SelfSLCompose(object):
    """
    Compose pre-processed data for Self-supervised learning (SSL).
    Through interval, how frequently SSL pipeline (pipeline1 + pipeline2) is applied is set.
    """
    def __init__(self, pipeline1, pipeline2):
        self.pipeline1 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline1])
        self.pipeline2 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline2])

    def __call__(self, data):
        data1 = self.pipeline1(deepcopy(data))
        #h1, w1, _ = data1['img_metas'].data['img_shape']
        #self.pipeline2.transforms[1].img_scale = [(w1, h1)]
        
        data2 = self.pipeline2(deepcopy(data))        
        
        data = deepcopy(data1)
        data['img'] = (data1['img'], data2['img'])
        data['img_metas'] = (data1['img_metas'], data2['img_metas'])
        data['gt_bboxes'] = (data1['gt_bboxes'], data2['gt_bboxes'])
        data['gt_labels'] = (data1['gt_labels'], data2['gt_labels'])

        return data