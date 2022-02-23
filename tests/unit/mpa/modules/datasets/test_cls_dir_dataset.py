import unittest
import os
import pytest
import numpy as np

from mpa.modules.datasets.cls_dir_dataset import ClsDirDataset
from mmcls.datasets import build_dataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClsDirDataset(unittest.TestCase):
    def setUp(self):
        """
        Classification Dataset from Directory settings
        """
        self.assets_path = 'tests/assets'

        self.dataset_type = 'ClsDirDataset'
        self.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.pipeline = [
            dict(type='Resize', size=(224, 224)),
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(type='Normalize', **self.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
        self.data_prefix = os.path.join(self.assets_path, 'dirs/classification')
        self.train_prefix = os.path.join(self.data_prefix, 'train')

        self.data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.train_prefix,
            pipeline=self.pipeline,
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_cls_dir_dataset(self):
        """
        Verifies that ClsDirDataset build works
        """
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(isinstance(dataset, ClsDirDataset))

        self.data_cfg['data_dir'] = 'path/not/'
        with pytest.raises(Exception):
            dataset = build_dataset(self.data_cfg)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_dir_dataset(self):
        """
        Verifies that ClsDirDataset config vary
        """
        dataset = build_dataset(self.data_cfg)
        classes = dataset.CLASSES
        self.assertTrue(isinstance(classes, list))
        self.assertEqual(len(classes), 3)
        self.assertEqual(len(dataset), 12)
        data_infos = dataset.data_infos
        self.assertTrue(isinstance(data_infos, list))
        self.assertTrue(isinstance(data_infos[0], dict))
        self.assertTrue(isinstance(data_infos[0]['gt_label'], np.ndarray))
        self.assertTrue(dataset[0].get('img_metas', False))

        self.assertEqual(dataset.num_classes, 3)
        self.assertEqual(dataset.samples_per_gpu, 1)
        self.assertEqual(dataset.workers_per_gpu, 1)

        self.assertTrue(dataset.use_labels)
        self.assertFalse(dataset.class_acc)

        # Use unlabeled data settings
        self.data_cfg['use_labels'] = False
        self.data_cfg['data_dir'] = os.path.join(self.data_prefix, 'unlabeled')
        dataset = build_dataset(self.data_cfg)
        self.assertEqual(dataset.num_classes, 1)
        self.assertEqual(dataset.CLASSES[0], 'unlabeled')

        # Use partial classes (2 of 3)
        classes = ['0', '1']
        data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.train_prefix,
            pipeline=self.pipeline,
            classes=classes
        )
        dataset = build_dataset(data_cfg)
        self.assertEqual(dataset.num_classes, len(classes))
        self.assertEqual(dataset.CLASSES, classes)

        # Incremental-Learning Settings
        new_classes = ['2']
        data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.train_prefix,
            pipeline=self.pipeline,
            new_classes=new_classes,
            classes=classes + new_classes
        )
        dataset = build_dataset(data_cfg)
        self.assertEqual(dataset.num_classes, len(classes) + len(new_classes))
        self.assertTrue(isinstance(dataset.img_indices, dict))
        self.assertTrue(isinstance(dataset.img_indices['old'], list))
        self.assertEqual(len(dataset.img_indices['old']), 8)
        self.assertTrue(isinstance(dataset.img_indices['new'], list))
        self.assertEqual(len(dataset.img_indices['new']), 4)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_dir_dataset_evaluation(self):
        """
        Verifies that ClsDirDataset evaluation function works
        """
        classes = ['0', '1', '2']
        new_classes = ['2']
        data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.train_prefix,
            pipeline=self.pipeline,
            new_classes=new_classes,
            classes=classes
        )

        dataset = build_dataset(data_cfg)
        np.random.seed(1234)
        temp_prob = np.random.rand(12, 4)
        eval_results = dataset.evaluate(temp_prob)
        self.assertTrue(isinstance(eval_results, dict))
        self.assertEqual(len(eval_results), 1)

        eval_results = dataset.evaluate(temp_prob, metric='class_accuracy')
        self.assertTrue(isinstance(eval_results, dict))
        self.assertEqual(len(eval_results), 4)
        eval_results = dataset.evaluate(temp_prob, metric=['accuracy', 'class_accuracy'])
        self.assertTrue(isinstance(eval_results, dict))
        self.assertEqual(len(eval_results), 5)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_dir_dataset_dict_pipeline(self):
        """
        Verifies that ClsDirDataset works with dictionary pipeline
        """
        strong_pipeline = [
            dict(type='Resize', size=(224, 224)),
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(type="RandomRotate", p=0.35, angle=(-10, 10)),
            dict(type="ToNumpy"),
            dict(type='Normalize', **self.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
        data_cfg = dict(
            type=self.dataset_type,
            data_dir=self.train_prefix,
            pipeline=dict(weak=self.pipeline, strong=strong_pipeline),
        )

        dataset = build_dataset(data_cfg)
        self.assertEqual(dataset.num_pipes, 2)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_dir_dataset_with_value_error(self):
        """
        Verifies that ClsDirDataset raise Exception
        use_label=True, but if the folder structure is not suitable (unlabeled)
        """
        self.data_cfg['data_dir'] = os.path.join(self.data_prefix, 'unlabeled')
        with self.assertRaises(ValueError):
            build_dataset(self.data_cfg)
