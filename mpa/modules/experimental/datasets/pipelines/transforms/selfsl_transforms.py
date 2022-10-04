import collections
import numpy as np
from PIL import Image

from torchvision import transforms as _transforms
from torchvision.transforms import functional as F

from mmdet.datasets import PIPELINES
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

@PIPELINES.register_module()
class ProbCompose(object):
    def __init__(self, transforms, probs):
        assert isinstance(transforms, collections.abc.Sequence)
        assert isinstance(probs, collections.abc.Sequence)
        assert len(transforms) == len(probs)
        assert all(p >= 0.0 for p in probs)

        sum_probs = float(sum(probs))
        assert sum_probs > 0.0
        norm_probs = [float(p) / sum_probs for p in probs]
        self.limits = np.cumsum([0.0] + norm_probs)

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        rand_value = np.random.rand()
        transform_id = np.max(np.where(rand_value > self.limits)[0])

        transform = self.transforms[transform_id]
        data = transform(data)

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return 