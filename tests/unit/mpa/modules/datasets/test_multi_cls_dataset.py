import unittest
import os
import pytest
import numpy as np

from mpa.modules.datasets.multi_cls_dataset import MultiClsDataset
from mmcls.datasets import build_dataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestMultiClassDataset(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

        dataset_type = 'MultiClsDataset'
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        pipeline = [
            dict(type='Resize', size=(224, 224)),
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
        data_prefix = os.path.join(self.assets_path, 'csvs/imgs')
        self.data_file = os.path.join(self.assets_path, 'csvs/dss18.data.csv')
        self.anno_file = os.path.join(self.assets_path, 'csvs/dss18.anno.multi_task.csv')
        self.tasks = {
            'Gender': ["Male", "Female", "Unknown"],
            'Backpack': ['Yes', 'No']
        }
        self.data_cfg = dict(
            type=dataset_type,
            data_prefix=data_prefix,
            ann_file=self.anno_file,
            data_file=self.data_file,
            pipeline=pipeline,
            tasks=self.tasks
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_dataset(self):
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(isinstance(dataset, MultiClsDataset))

        self.data_cfg['data_file'] = 'path/not/exist.csv'
        with pytest.raises(Exception):
            dataset = build_dataset(self.data_cfg)
        # self.assertEqual(e.type, ValueError)

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
        dummy_res = [{'Gender': np.array([[1., 0., 0.]]),
                      'Backpack': np.array([[1., 0.]])} for i in range(10)]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'class_accuracy']
        for metric in metrics:
            eval = dataset.evaluate(dummy_res, metric)
            self.assertTrue(isinstance(eval, dict))
