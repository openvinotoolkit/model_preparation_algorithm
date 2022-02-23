import unittest
import pytest
import torchvision
from PIL import Image

from mpa.modules.datasets.pipelines.transforms.augmix import (
    AugMixAugment, OpsFabric, _AUGMIX_TRANSFORMS, _AUGMIX_TRANSFORMS_GREY
)

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestAugmix(unittest.TestCase):
    def setUp(self):
        self.dataset = getattr(torchvision.datasets, "SVHN")(
            "./data/torchvision/svhn", split="train", download=True
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_augmix(self):
        """
        Verifies that AugMix works & all augment_pool is working
        """
        augmix = AugMixAugment(config_str="augmix-m5-w3")
        img = self.dataset[0][0]
        dummy_data = {"img": img}
        result = augmix(dummy_data)
        self.assertTrue(result["augmix"])
        self.assertEqual(len(augmix.ops), 12)
        self.assertEqual(augmix.width, 3)

        augmix = AugMixAugment(config_str="augmix-m5-w5")
        self.assertEqual(augmix.width, 5)

        # Set grey parameter
        augmix = AugMixAugment(config_str="augmix-m5-w5", grey=True)
        self.assertEqual(len(augmix.ops), 5)

        # all op test
        hparams = dict(
            translate_const=250,
            img_mean=tuple([int(c * 256) for c in [0.485, 0.456, 0.406]]),
            magnitude_std=float("inf"),
        )
        op_lst = [OpsFabric(name, 3, hparams, 1.0) for name in _AUGMIX_TRANSFORMS]
        for op in op_lst:
            img_aug = op(img)
            self.assertTrue(isinstance(img_aug, Image.Image))
        op_lst = [OpsFabric(name, 3, hparams, 1.0) for name in _AUGMIX_TRANSFORMS_GREY]
        for op in op_lst:
            img_aug = op(img)
            self.assertTrue(isinstance(img_aug, Image.Image))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_augmix_wrong_parameter(self):
        with self.assertRaises(AssertionError):
            AugMixAugment(config_str="m5-w3")
        with self.assertRaises(AssertionError):
            AugMixAugment(config_str="augmix-z5")
        with self.assertRaises(ValueError):
            AugMixAugment(config_str="augmix-m0.1")
