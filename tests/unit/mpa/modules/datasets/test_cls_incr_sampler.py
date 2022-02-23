import unittest
import pytest
import numpy as np

from mmcls.datasets import build_dataset
from mpa.modules.datasets.samplers.cls_incr_sampler import ClsIncrSampler

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClsIncrSampler(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'
        dataset_type = 'ClsTVDataset'
        self.data_cfg = dict(
            type=dataset_type,
            base='CIFAR10',
            num_images=32,
            classes=[0, 1, 2, 3],
            new_classes=[3]
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_incr_sampler(self):
        # Default Settings
        dataset = build_dataset(self.data_cfg)
        batch_size = 8
        sampler = ClsIncrSampler(dataset, batch_size)
        self.assertEqual(len(sampler), dataset.num_images)
        self.assertEqual(len(sampler.new_indices) + len(sampler.old_indices), dataset.num_images)
        self.assertEqual(len(sampler.new_indices), 8)
        self.assertEqual(len(sampler.old_indices), 24)
        old_new_ratio = np.sqrt(len(sampler.old_indices) / len(sampler.new_indices))
        self.assertEqual(sampler.old_new_ratio, int(old_new_ratio))

        # Repeat = 3
        dataset = build_dataset(self.data_cfg)
        dataset.times = 3
        sampler = ClsIncrSampler(dataset, batch_size)
        self.assertEqual(len(sampler), dataset.num_images * dataset.times)

        # Efficient Mode
        dataset = build_dataset(self.data_cfg)
        batch_size = 8
        sampler = ClsIncrSampler(dataset, batch_size, efficient_mode=True)
        self.assertEqual(len(sampler.new_indices) + len(sampler.old_indices), dataset.num_images)
        self.assertEqual(len(sampler.new_indices), 8)
        self.assertEqual(len(sampler.old_indices), 24)
        old_new_ratio = np.sqrt(len(sampler.old_indices) / len(sampler.new_indices))
        data_length = int(len(sampler.new_indices) * (1 + old_new_ratio))
        self.assertEqual(len(sampler), data_length)
        self.assertEqual(sampler.old_new_ratio, 1)

        # no new_classes
        data_cfg = dict(
            type='ClsTVDataset',
            base='CIFAR10',
            num_images=32,
            classes=[0, 1, 2, 3],
        )
        dataset = build_dataset(data_cfg)
        with self.assertRaises(ValueError):
            sampler = ClsIncrSampler(dataset, batch_size)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_iter_cls_incr_sampler(self):
        data_cfg = dict(
            type='ClsTVDataset',
            base='CIFAR10',
            num_images=40,
            classes=[0, 1, 2, 3, 4],
            new_classes=[4]
        )
        dataset = build_dataset(data_cfg)
        batch_size = 8
        sampler = ClsIncrSampler(dataset, batch_size)
        iter_sampler = iter(sampler)
        self.assertEqual(len(list(iter_sampler)), dataset.num_images)
