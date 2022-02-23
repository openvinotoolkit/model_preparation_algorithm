import pytest
import os
import numpy as np
import torch
import copy
import cv2 as cv
from torchvision import transforms as T
from PIL import Image, ImageFilter
from mpa.modules.datasets.pipelines.torchvision2mmdet import (
    ColorJitter, RandomGrayscale, RandomErasing, RandomGaussianBlur, RandomApply
)

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_color_jitter():
    img = torch.rand(1, 3, 5, 5)
    inputs = {'img': copy.deepcopy(img)}
    tv_op = T.ColorJitter()
    md_op = ColorJitter()
    tv_output = tv_op(img)
    md_output = md_op(inputs)
    np.testing.assert_array_equal(tv_output, md_output['img'])

@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_random_gray_scale():
    img = torch.rand(1, 3, 5, 5)
    inputs = {'img': copy.deepcopy(img)}
    tv_op = T.RandomGrayscale(p=1.0)
    md_op = RandomGrayscale(p=1.0)
    tv_output = tv_op(img)
    md_output = md_op(inputs)
    np.testing.assert_array_equal(tv_output, md_output['img'])

@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_random_erasing():
    img = torch.rand(1, 3, 5, 5)
    inputs = {'img': copy.deepcopy(img)}
    md_op = RandomErasing(p=1.0)
    md_output = md_op(inputs)
    assert not np.array_equal(img, md_output['img'])

@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_random_gaussian_blur():
    img = np.random.rand(3, 5, 5)
    img = Image.fromarray(img, mode='RGB')
    inputs = {'img': copy.deepcopy(img)}
    output = img.filter(ImageFilter.GaussianBlur(radius=0.1))
    md_op = RandomGaussianBlur(0.1, 0.1)
    md_output = md_op(inputs)
    np.testing.assert_array_equal(
        np.asarray(output),
        np.asarray(md_output['img'])
    )

@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_random_random_gaussian_blur():
    img = np.random.rand(3, 5, 5)
    inputs = {'img': copy.deepcopy(img)}
    img = Image.fromarray(img, mode='RGB')
    inputs = {'img': copy.deepcopy(img)}
    output = img.filter(ImageFilter.GaussianBlur(radius=0.1))
    md_op = RandomApply(
        [dict(
            type='RandomGaussianBlur',
            sigma_min=0.1,
            sigma_max=0.1
        )],
        p=1.0
    )
    md_output = md_op(inputs)
    np.testing.assert_array_equal(
        np.asarray(output),
        np.asarray(md_output['img'])
    )

