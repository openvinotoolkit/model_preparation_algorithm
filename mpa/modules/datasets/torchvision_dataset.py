import inspect
import numpy as np
import PIL.Image as Image
# import torch
import torchvision.datasets as _datasets
from mmcv.utils import Registry, build_from_cfg

from mmcls.datasets import BaseDataset
from mmcls.datasets import DATASETS


# register existing datasets in torchvision
_DATASETS = Registry('torchvision_datasets')
for m in inspect.getmembers(_datasets, inspect.isclass):
    _DATASETS.register_module(module=m[1])


@DATASETS.register_module()
class TorchVisionDataset(BaseDataset):
    """Wrapper of TorchVision Datasets
    """
    def __init__(self, data_prefix, dataset, pipeline=[], data_file=None,
                 **kwargs):
        dataset['root'] = data_prefix
        self.dataset = build_from_cfg(dataset, _DATASETS)
        super(TorchVisionDataset, self).__init__(data_prefix, pipeline,
                                                 **kwargs)

    def load_annotations(self):
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, gt = self.dataset[idx]
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError('wrong type - {}'.format(type(img)))
        results = {'img': img, 'gt_label': np.array(gt)}
        return self.pipeline(results)

    def get_gt_labels(self):
        gt_labels = np.array([gt for _, gt in self.dataset])
        return gt_labels
