import numpy as np
from PIL import Image

from torchvision import transforms as _transforms
from torchvision.transforms import functional as F

from mmseg.datasets import PIPELINES
from mmcv.utils import build_from_cfg


@PIPELINES.register_module
class RandomResizedCrop(_transforms.RandomResizedCrop):
    def __call__(self, results):
        img = Image.fromarray(results['img'])

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = np.array(F.resized_crop(img, i, j, h, w, self.size, self.interpolation))
        results['img'] = img
        results['img_shape'] = img.shape
        for key in results.get('seg_fields', []):
            results[key] = np.array(
                F.resized_crop(
                    Image.fromarray(results[key]), 
                    i, j, h, w, self.size, self.interpolation))

        return results


@PIPELINES.register_module
class ColorJitter(_transforms.ColorJitter):
    def __call__(self, results):
        results['img'] = np.array(self.forward(Image.fromarray(results['img'])))
        return results


@PIPELINES.register_module
class RandomGrayscale(_transforms.RandomGrayscale):
    def __call__(self, results):
        results['img'] = np.array(self.forward(Image.fromarray(results['img'])))
        return results


@PIPELINES.register_module
class GaussianBlur(_transforms.GaussianBlur):
    def __call__(self, results):
        results['img'] = np.array(self.forward(Image.fromarray(results['img']))) 
        return results


@PIPELINES.register_module
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, results):
        img = results['img']
        img = np.where(img < self.threshold, img, 255-img)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
