import numpy as np
import torch

from .builder import DATASOURCES


@DATASOURCES.register_module()
class ImageIndexList(object):

    def __init__(self, base, indices, **kwargs):
        self.base = base
        self.data = base.data[indices]
        if isinstance(self.data, torch.Tensor):
            # It is expected that np.array will be included in self.data
            # but in the case of FashionMNIST, Tensor is included, causing an error
            self.data = [np.array(base[idx][0].convert("RGB")) for idx in indices]
        if hasattr(base, 'targets'):
            self.targets = np.array(base.targets)[indices]
        elif hasattr(base, 'labels'):
            self.targets = np.array(base.labels)[indices]
        else:
            raise NotImplementedError(f'check the attribute name of torchvision dataset {base}')

    def get_length(self):
        return len(self.data)

    def get_sample(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # TODO: need to implement auto-reshaping img arrary
        if len(np.shape(img)) == 3 and np.shape(img)[2] != 3:  # img should have HWC shape
            img = np.transpose(img, (1, 2, 0))
        return img, target
