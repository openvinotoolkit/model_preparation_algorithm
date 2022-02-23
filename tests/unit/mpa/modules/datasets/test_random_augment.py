import unittest
import pytest
import torchvision
import random
import numpy as np

from mpa.modules.datasets.pipelines.transforms.random_augment import RandAugment

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestRandAugment(unittest.TestCase):
    def setUp(self):
        """
        setting svhn dataset for augmentation test
        """
        self.dataset = getattr(torchvision.datasets, "SVHN")(
            "./data/torchvision/svhn", split="train", download=True
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_random_augment(self):
        """
        Verifies that RandAugment works & all augment_pool is working
        """
        random.seed(1234)
        np.random.seed(1234)
        rand_aug = RandAugment(n=2, m=10)
        img = self.dataset[0][0]
        dummy_data = {"img": img}
        result = rand_aug(dummy_data)

        self.assertTrue(isinstance(result, dict))
        self.assertTrue(result["CutoutAbs"])

        rand_aug_pool = rand_aug.augment_pool
        self.assertEqual(len(rand_aug_pool), 14)
        pool_returned = [
            None, 0.77, 0.68, 0.86, None, None, 4,
            21, 0.32, 0.03, -0.18, 76, 6, -3
        ]
        for i, aug in enumerate(rand_aug_pool):
            op, max_v, bias = aug
            v = np.random.randint(1, 10)
            _, aug_v = op(img, v=v, max_v=max_v, bias=bias)
            self.assertEqual(aug_v, pool_returned[i])
