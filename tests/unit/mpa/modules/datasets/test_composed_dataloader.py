import unittest
import os
import torch
import pytest

from mpa.modules.datasets.composed_dataloader import ComposedDL, CDLIterator
from mmcls.datasets import build_dataset, build_dataloader

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestComposedDL(unittest.TestCase):
    def setUp(self):
        """
        ComposedDL settings with ClsDirDataset Settings
        """
        self.assets_path = 'tests/assets'

        self.dataset_type = 'ClsDirDataset'
        self.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.pipeline = [
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(type='Normalize', **self.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
        self.data_prefix = os.path.join(self.assets_path, 'dirs/classification')
        self.labeled_prefix = os.path.join(self.data_prefix, 'train')
        self.unlabeled_prefix = os.path.join(self.data_prefix, 'unlabeled')

    def tearDown(self):
        pass

    def get_sub_loaders(self, labeled_batch_size, unlabeled_batch_size):
        label_data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.labeled_prefix,
            pipeline=self.pipeline,
            samples_per_gpu=labeled_batch_size,
        )

        unlabel_data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.unlabeled_prefix,
            pipeline=dict(weak=self.pipeline, strong=self.pipeline),
            use_labels=False,
            samples_per_gpu=unlabeled_batch_size
        )
        ds = [build_dataset(label_data_cfg), build_dataset(unlabel_data_cfg)]
        sub_loaders = [
            build_dataloader(
                sub_ds,
                sub_ds.samples_per_gpu if hasattr(sub_ds, 'samples_per_gpu') else 1,
                1,
                num_gpus=1,
                dist=False,
                round_up=True,
                seed=1234,
            ) for sub_ds in ds
        ]
        return sub_loaders

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_composed_dl(self):
        """
        Verifies that ComposedDL works
        """
        torch.manual_seed(1234)
        labeled_bs = 3
        unlabeled_bs = 4
        composed_dl = ComposedDL(self.get_sub_loaders(labeled_bs, unlabeled_bs))
        self.assertEqual(composed_dl.max_iter, 4)
        self.assertEqual(len(composed_dl), 4)

        composed_iter = iter(composed_dl)
        self.assertTrue(isinstance(composed_iter, CDLIterator))

        for i in range(len(composed_dl)):
            batch = next(composed_iter)
            self.assertTrue(isinstance(batch, dict))
            self.assertEqual(len(batch['img']), labeled_bs)
            self.assertEqual(len(batch['gt_label']), labeled_bs)

            self.assertIn('extra_0', batch)
            self.assertTrue(isinstance(batch['extra_0'], dict))
            self.assertIn('weak', batch['extra_0'])
            self.assertEqual(len(batch['extra_0']['weak']['img']), unlabeled_bs)
            self.assertIn('strong', batch['extra_0'])
            self.assertEqual(len(batch['extra_0']['strong']['img']), unlabeled_bs)

        # over max-iteration
        with self.assertRaises(StopIteration):
            batch = next(composed_iter)

        labeled_bs = 3
        unlabeled_bs = 5  # unlabeled data: 4 images
        composed_dl = ComposedDL(self.get_sub_loaders(labeled_bs, unlabeled_bs))
        composed_iter = iter(composed_dl)
        batch = next(composed_iter)
        self.assertEqual(len(batch['img']), labeled_bs)
        self.assertEqual(len(batch['gt_label']), labeled_bs)
        self.assertEqual(len(batch['extra_0']['weak']['img']), 4)
        self.assertEqual(len(batch['extra_0']['strong']['img']), 4)

        labeled_bs = 16  # labeled data: 12 images
        unlabeled_bs = 4
        composed_dl = ComposedDL(self.get_sub_loaders(labeled_bs, unlabeled_bs))
        composed_iter = iter(composed_dl)
        batch = next(composed_iter)
        self.assertEqual(len(batch['img']), 12)
        self.assertEqual(len(batch['gt_label']), 12)
        self.assertEqual(len(batch['extra_0']['weak']['img']), unlabeled_bs)
        self.assertEqual(len(batch['extra_0']['strong']['img']), unlabeled_bs)
