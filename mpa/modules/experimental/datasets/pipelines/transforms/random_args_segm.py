# flake8: noqa
# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

from mmseg.datasets.builder import PIPELINES

PARAMETER_MAX = 10


def Identity(img, **kwarg):
    return img


def AutoContrast(img, gt, **kwarg):
    return PIL.ImageOps.autocontrast(img), Identity(gt), None


def Brightness(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v), Identity(gt), v


def Color(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v), Identity(gt), v


def Contrast(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v), Identity(gt), v


def Equalize(img, gt, **kwarg):
    return PIL.ImageOps.equalize(img), Identity(gt), None


def Invert(img, gt, **kwarg):
    return PIL.ImageOps.invert(img), Identity(gt), None


def Posterize(img, gt, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v), Identity(gt), v


def Rotate(img, gt, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v), gt.rotate(v), v


def Sharpness(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v), Identity(gt), v


def ShearX(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), gt.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), v


def ShearY(img, gt, v, max_v, bias=0):
    # print(gt)
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), gt.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), v


def Solarize(img, gt, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v), Identity(gt), v


def SolarizeAdd(img, gt, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), Identity(gt), v


def TranslateX(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), gt.transform(gt.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), v


def TranslateY(img, gt, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), gt.transform(gt.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), v


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)
            ]
    return augs


@PIPELINES.register_module()
class RandAugmentSemiSeg(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, results):
        # import time
        # now = time.time()
        img = results['img']
        gt = results['gt_semantic_seg']
        if not Image.isImageType(img):
            img = Image.fromarray(results['img'])
        if not Image.isImageType(gt):
            gt = Image.fromarray(results['gt_semantic_seg'], 'L')
        # img.save(f'/media/hdd2/jeom/mpa/tmp/img_{now}.png')
        # gt.save(f'/media/hdd2/jeom/mpa/tmp/gt_{now}.png')
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img, gt, v = op(img, gt, v=v, max_v=max_v, bias=bias)
                results['rand_semiseg_img_{}'.format(op.__name__)] = v
                results['rand_semiseg_gt_{}'.format(op.__name__)] = v
        # img.save(f'/media/hdd2/jeom/mpa/tmp/img_randaug_{now}.png')
        # gt.save(f'/media/hdd2/jeom/mpa/tmp/gt_randaug_{now}.png')
        results['img'] = np.array(img)
        results['gt_semantic_seg'] = np.array(gt)
        return results
