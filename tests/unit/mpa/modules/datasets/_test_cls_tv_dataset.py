import os
import unittest
from unittest.mock import patch, Mock, MagicMock
import pytest
import numpy as np

from torch import Tensor
from torchvision.datasets import CIFAR10
from mmcls.datasets import build_dataset
from mpa.modules.experimental.datasets.cls_tv_dataset import ClsTVDataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClsTVDataset(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'
        dataset_type = 'ClsTVDataset'
        img_norm_cfg = dict(
            mean=[129.3, 124.1, 112.4], std=[68.2,  65.4,  70.4], to_rgb=True)
        self.train_pipeline = [
            dict(type='Resize', size=(224, 224)),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
        self.data_cfg = dict(
            type=dataset_type,
            base='CIFAR10',
            train=True,
            pipeline=self.train_pipeline,
            samples_per_gpu=16,
            workers_per_gpu=4,
            download=True
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_dataset(self):
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(isinstance(dataset, ClsTVDataset))

        self.data_cfg['base'] = 'CIFAR1000'
        with pytest.raises(Exception) as e:
            dataset = build_dataset(self.data_cfg)
        self.assertEqual(e.type, AttributeError)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_dataset(self):
        dataset = build_dataset(self.data_cfg)
        data_infos = dataset.data_infos
        self.assertTrue(isinstance(data_infos, list))
        self.assertTrue(isinstance(data_infos[0], dict))
        self.assertTrue(isinstance(data_infos[0]['gt_label'], np.ndarray))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_evaluate(self):
        dataset = build_dataset(self.data_cfg)
        ds_len = len(dataset)
        dummy_res = [np.array([[1., 0., 0.]]) for i in range(ds_len)]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            eval = dataset.evaluate(dummy_res, metric)
            self.assertTrue(isinstance(eval, dict))
            for k, v in eval.items():
                self.assertTrue(k.startswith(metric))

        # class_accuracy
        metric = 'class_accuracy'
        eval = dataset.evaluate(dummy_res, metric)
        self.assertTrue(isinstance(eval, dict))
        for k, v in eval.items():
            self.assertTrue(k.endswith('accuracy'))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_base_dataset(self):
        bases = ['CIFAR10', 'CIFAR100', 'STL10', 'SVHN']
        num_images = 10
        for base in bases:
            ds = ClsTVDataset(
                base,
                num_images=num_images,
                balanced_class=False,
            )
            item = ds[0]
            self.assertEqual(num_images, len(ds))
            self.assertTrue('img' in item)
            self.assertIsInstance(item['img'], Tensor)
            self.assertTrue('gt_label' in item)
            self.assertIsInstance(item['gt_label'], Tensor)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_dataset_init(self):
        base = Mock()
        base.targets = []
        base.labels = []
        delattr(base, 'targets')
        delattr(base, 'labels')
        with pytest.raises(Exception) as e:
            ds = ClsTVDataset(base)
        self.assertEqual(e.type, NotImplementedError)

        num_indices = 10
        baseset_len = 50000
        base = CIFAR10('data/torchvision/cifar10', download=True)
        ds = ClsTVDataset(base)
        ds2 = ClsTVDataset('CIFAR10')
        self.assertEqual(len(ds), len(ds2))

        with pytest.raises(Exception) as e:
            ds = ClsTVDataset(base, include_idx=range(num_indices))
        self.assertEqual(e.type, TypeError)

        ds = ClsTVDataset(base, include_idx=list(range(num_indices)))
        self.assertEqual(len(ds), num_indices)

        # test for include_idx param
        ds = ClsTVDataset(base, include_idx=os.path.join(self.assets_path, 'indices/cifar10-7k.p'))
        self.assertEqual(len(ds), 7000)

        logger_warning = MagicMock()
        with patch('mpa.modules.datasets.cls_tv_dataset.logger.warning', logger_warning):
            ds = ClsTVDataset(base, include_idx='path/not/exist.p')
        logger_warning.assert_called()
        del logger_warning

        # test for exclude_idx param
        with pytest.raises(Exception) as e:
            ds = ClsTVDataset(base, exclude_idx=range(num_indices))
        self.assertEqual(e.type, TypeError)

        ds = ClsTVDataset(base, exclude_idx=list(range(num_indices)))
        self.assertEqual(len(ds), baseset_len - num_indices)

        ds = ClsTVDataset(base, exclude_idx=os.path.join(self.assets_path, 'indices/cifar10-7k.p'))
        self.assertEqual(len(ds), baseset_len - 7000)

        logger_warning = MagicMock()
        with patch('mpa.modules.datasets.cls_tv_dataset.logger.warning', logger_warning):
            ds = ClsTVDataset(base, exclude_idx='path/not/exist.p')
        logger_warning.assert_called()

        # test num_images param
        with pytest.raises(Exception) as e:
            ds = ClsTVDataset(base, num_images=baseset_len+1)
        self.assertEqual(e.type, RuntimeError)

        with pytest.raises(Exception) as e:
            ds = ClsTVDataset(base, num_images=-100)
        self.assertEqual(e.type, RuntimeError)

        split_len = 100
        ds = ClsTVDataset(base, num_images=split_len)
        self.assertEqual(len(ds), split_len)

        ds = ClsTVDataset(base, num_images=split_len, use_labels=False)
        self.assertEqual(len(ds), split_len)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_dataset_classes(self):
        ds = ClsTVDataset(
            'CIFAR10',
        )
        self.assertEqual(len(ds.CLASSES), 10)

        ds = ClsTVDataset(
            'CIFAR10',
            classes=[0, 1, 2]
        )
        self.assertEqual(len(ds.CLASSES), 3)

        ds = ClsTVDataset(
            'CIFAR10',
            classes=[0, 1, 2]
        )
        self.assertEqual(len(ds.CLASSES), 3)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_incr_data(self):
        base = 'CIFAR10'
        ds = ClsTVDataset(
            base,
            num_images=40,
            classes=[0, 1, 2, 3],
            new_classes=[3]
        )
        self.assertEqual(len(ds.img_indices['old']), 30)
        self.assertEqual(len(ds.img_indices['new']), 10)
