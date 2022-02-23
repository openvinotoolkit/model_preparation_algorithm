import unittest
import pytest
import torchvision
import torch
from PIL import Image

from mpa.modules.datasets.pipelines.transforms.ote_transforms import (
    RandomRotate, PILToTensor, TensorNormalize
)

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestOTETransforms(unittest.TestCase):
    def setUp(self):
        self.dataset = getattr(torchvision.datasets, 'SVHN')('./data/torchvision/svhn', split='train', download=True)

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_random_rotate(self):
        random_rotate = RandomRotate(p=0.35, angle=(-10, 10))
        img = self.dataset[0][0]
        dummy_data = {
            'img': img
        }
        result = random_rotate(dummy_data)
        while 'RandomRotate' not in result:
            result = random_rotate(dummy_data)
        self.assertTrue(result['RandomRotate'])
        self.assertIsInstance(result['img'], Image.Image)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_pil_to_tensor(self):
        pil_to_tensor = PILToTensor()
        img = self.dataset[0][0]
        dummy_data = {
            'img': img
        }
        self.assertIsInstance(img, Image.Image)
        result = pil_to_tensor(dummy_data)
        self.assertIsInstance(result['img'], torch.Tensor)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_tensor_normalize(self):
        __img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor_normalize = TensorNormalize(**__img_norm_cfg)
        img = torch.ones((3, 3, 3))
        dummy_data = {
            'img': img
        }
        result = tensor_normalize(dummy_data)
        self.assertIsInstance(result['img'], torch.Tensor)
        self.assertTrue(result['TensorNormalize'])
        mean = torch.as_tensor(__img_norm_cfg['mean'], dtype=img.dtype, device=img.device)
        std = torch.as_tensor(__img_norm_cfg['std'], dtype=img.dtype, device=img.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        expected = img.sub_(mean).div_(std)
        self.assertTrue(torch.equal(result['img'], expected))
