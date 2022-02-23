import os.path as osp
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mpa.selfsl.builder import DATASETS
from mpa.selfsl.builder import build_datasource, build_pipeline


@DATASETS.register_module()
class SelfSLDataset(Dataset):
    """Wrapper of existing dataset for SelfSL
    """
    def __init__(self, down_task, datasource, pipeline):
        self.datasource = build_datasource(datasource, down_task)

        pipeline1 = [build_pipeline(p) for p in pipeline['view0']]
        self.pipeline1 = Compose(pipeline1)
        pipeline2 = [build_pipeline(p) for p in pipeline['view1']]
        self.pipeline2 = Compose(pipeline2)

    def __len__(self):
        return len(self.datasource)

    def _getitem_from_source(self, idx):
        data = self.datasource[idx]
        if isinstance(data, dict):  # dict(img:ndarray)
            if 'img' not in data.keys():
                if data['img_prefix'] is not None:
                    filename = osp.join(data['img_prefix'],
                                        data['img_info']['filename'])
                else:
                    filename = data['img_info']['filename']
                data['filename'] = filename
                data['img'] = Image.open(filename)
            img = data['img']
        elif isinstance(data, tuple):  # (img:[ndarray|Image], gt)
            img, _ = data
        else:
            img = data

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError('image should be PIL.Image or numpy.ndarray \
                 - {}'.format(type(img)))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self._getitem_from_source(idx)

        results1 = self.pipeline1(dict(img=img))
        results2 = self.pipeline2(dict(img=img))

        results = dict()
        for k, v in results1.items():
            results[k+'1'] = v
        for k, v in results2.items():
            results[k+'2'] = v

        return results

    def evaluate(self, **kwargs):
        pass
