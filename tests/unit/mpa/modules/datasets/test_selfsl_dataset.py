# from inspect import CORO_CREATED
import unittest
import os
import pytest
# import numpy as np
from torch import Tensor

from mmcls.datasets import build_dataset as build_cls_dataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

from mpa.modules.datasets.selfsl_dataset import SelfSLDataset
from mpa.selfsl.builder import build_dataset


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSlDataset(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

        self.datasource_cfgs = dict(
            FashionMNIST=dict(
                type='FashionMNIST',
                data_prefix='data/torchvision/fmnist',
                pipeline=[]),
            TVDatasetSplit=dict(
                type='TVDatasetSplit',
                data_prefix='data/torchvision/cifar10',
                base='CIFAR10',
                pipeline=[],
                train=True,
                download=True),
            CSVDatasetCls=dict(
                type='CSVDatasetCls',
                data_prefix=os.path.join(self.assets_path, 'csvs/imgs'),
                ann_file=os.path.join(self.assets_path, 'csvs/dss18.anno.single_cls.csv'),
                data_file=os.path.join(self.assets_path, 'csvs/dss18.data.csv'),
                pipeline=[],
                classes=["Other", "Senior", "Kids", "Unknown"]),
            )

        dataset_type = 'SelfSLDataset'
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        pipeline = [
            dict(type='RandomResizedCrop', size=(224, 224)),
            dict(type='RandomHorizontalFlip'),
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='ColorJitter',
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1)
                ],
                p=0.8),
            dict(type='RandomGrayscale', p=0.2),
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='GaussianBlur',
                        sigma_min=0.1,
                        sigma_max=2.0)
                ],
                p=1.),
            dict(
                type='RandomAppliedTrans',
                transforms=[dict(type='Solarization')],
                p=0.),
            dict(type='ToNumpy'),
            dict(type='Normalize',  **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]

        self.data_cfg = dict(
            type=dataset_type,
            down_task='classification',
            datasource=self.datasource_cfgs['FashionMNIST'],
            pipeline=dict(
                view0=pipeline,
                view1=pipeline
            ),
        )

        self.pipeline_with_coord = [
            dict(type='RandomResizedCrop', size=(224, 224), with_coord=True),
            dict(type='RandomHorizontalFlip', with_coord=True),
            dict(type='ToNumpy'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'coord'])
        ]

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_dataset(self):
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(isinstance(dataset, SelfSLDataset))

        self.data_cfg['data_file'] = 'path/not/exist.csv'
        with pytest.raises(Exception):
            dataset = build_dataset(self.data_cfg)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_dataset(self):
        dataset = build_dataset(self.data_cfg)
        cls_dataset = build_cls_dataset(self.data_cfg['datasource'])
        self.assertEqual(len(dataset), len(cls_dataset))
        data = dataset[0]
        self.assertNotIn('coord1', data)
        self.assertNotIn('coord2', data)
        self.assertIsInstance(data['img1'], Tensor)
        self.assertIsInstance(data['img2'], Tensor)
        self.assertEqual(data['img1'].shape, (3, 224, 224))
        self.assertEqual(data['img2'].shape, (3, 224, 224))

        self.data_cfg['pipeline']['view0'] = self.pipeline_with_coord
        self.data_cfg['pipeline']['view1'] = self.pipeline_with_coord
        dataset = build_dataset(self.data_cfg)
        cls_dataset = build_cls_dataset(self.data_cfg['datasource'])
        self.assertEqual(len(dataset), len(cls_dataset))
        data = dataset[0]
        self.assertIn('img1', data)
        self.assertIn('img2', data)
        self.assertIsInstance(data['coord1'], Tensor)
        self.assertIsInstance(data['coord2'], Tensor)
        self.assertEqual(data['coord1'].shape, (4,))
        self.assertEqual(data['coord2'].shape, (4,))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_datasource(self):
        self.data_cfg['datasource'] = self.datasource_cfgs['CSVDatasetCls']
        self.data_cfg['datasource']['pipeline'] = [dict(type='ToPIL')]
        self.data_cfg['pipeline']['view0'][0]['size'] = 32    # Resize
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(len(dataset), 10)
        data = dataset[0]
        self.assertEqual(data['img1'].shape, (3, 32, 32))
        self.assertEqual(data['img2'].shape, (3, 32, 32))

        self.data_cfg['datasource'] = self.datasource_cfgs['TVDatasetSplit']
        self.data_cfg['datasource']['pipeline'] = [dict(type='ImageToTensor', keys=['img'])]
        dataset = build_dataset(self.data_cfg)
        with pytest.raises(TypeError):
            data = dataset[0]
