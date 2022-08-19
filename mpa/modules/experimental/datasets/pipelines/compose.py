from copy import deepcopy
import numpy as np

from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import Compose
from mmseg.datasets.pipelines import to_tensor
from mmcv.utils import build_from_cfg


@PIPELINES.register_module()
class SelfSLCompose(object):
    def __init__(self, pipeline1, pipeline2):
        self.pipeline1 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline1])
        self.pipeline2 = Compose([build_from_cfg(p, PIPELINES) for p in pipeline2])

    def __call__(self, data):
        data1 = self.pipeline1(deepcopy(data))
        data2 = self.pipeline2(deepcopy(data))

        data = deepcopy(data1)
        data['img'] = to_tensor(np.ascontiguousarray(np.stack((data1['img'], data2['img']), axis=0).transpose(0, 3, 1, 2)))
        data['gt_semantic_seg'] = to_tensor(
            np.stack((data1['gt_semantic_seg'], data2['gt_semantic_seg']), axis=0)[:, None, ...].astype(np.int64))
        data['flip'] = [data1['flip'], data2['flip']]

        return data
