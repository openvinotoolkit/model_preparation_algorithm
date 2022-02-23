import unittest
import os
import pytest
import numpy as np

from mpa.modules.datasets.cls_csv_incr_dataset import LwfTaskIncDataset, ClassIncDataset
from mmcls.datasets import build_dataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClsLwfDataset(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

        dataset_type = 'LwfTaskIncDataset'
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.pipeline = [
            dict(type='Resize', size=(224, 224)),
            dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
        self.data_prefix = os.path.join(self.assets_path, 'csvs/imgs')
        self.data_file = os.path.join(self.assets_path, 'csvs/dss18.data.csv')
        self.anno_file = os.path.join(self.assets_path, 'csvs/dss18.anno.multi_task.csv')
        self.tasks = {
            'Age': ["Other", "Senior", "Kids", "Unknown"]
        }
        self.data_cfg = dict(
            type=dataset_type,
            data_prefix=self.data_prefix,
            ann_file=self.anno_file,
            data_file=self.data_file,
            pipeline=self.pipeline,
            tasks=self.tasks
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_task_incr_dataset(self):
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(isinstance(dataset, LwfTaskIncDataset))

        self.data_cfg['data_file'] = 'path/not/exist.csv'
        with pytest.raises(Exception):
            dataset = build_dataset(self.data_cfg)
        # self.assertEqual(e.type, ValueError)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_task_incr_dataset(self):
        dataset = build_dataset(self.data_cfg)
        data_infos = dataset.data_infos
        self.assertTrue(isinstance(data_infos, list))
        self.assertTrue(isinstance(data_infos[0], dict))
        self.assertTrue(isinstance(data_infos[0]['gt_label'], np.ndarray))
        self.assertFalse(data_infos[0].get('soft_labels', False))

        old_tasks = {'dummy_task': ['1', '2', '3']}
        old_prob = {'dummy_task': np.random.rand(10, 3)}
        model_tasks = old_tasks
        model_tasks.update(self.tasks)
        for i, data_info in enumerate(data_infos):
            data_info['soft_label'] = {task: value[i] for task, value in old_prob.items()}
        np.save('dummy_task_inc.npy', data_infos)
        dataset_lwf = build_dataset(self.data_cfg,
                                    default_args={'pre_stage_res': 'dummy_task_inc.npy', 'model_tasks': model_tasks})
        data_infos = dataset_lwf.data_infos
        self.assertTrue(data_infos[0].get('soft_label', False))
        self.assertTrue(isinstance(data_infos[0]['soft_label'], dict))
        os.remove('dummy_task_inc.npy')

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_cls_incr_dataset(self):
        dataset_type = 'ClassIncDataset'
        data_file = os.path.join(self.assets_path, 'csvs/dss18.data.csv')
        anno_file = os.path.join(self.assets_path, 'csvs/dss18.anno.single_cls.csv')
        classes = ["Other", "Senior", "Kids", "Unknown"]
        self.data_cfg = dict(
            type=dataset_type,
            data_prefix=self.data_prefix,
            ann_file=anno_file,
            data_file=data_file,
            pipeline=self.pipeline,
            classes=classes
        )
        dataset = build_dataset(self.data_cfg)
        self.assertTrue(isinstance(dataset, ClassIncDataset))

        self.data_cfg['data_file'] = 'path/not/exist.csv'
        with pytest.raises(Exception):
            dataset = build_dataset(self.data_cfg)
        # self.assertEqual(e.type, ValueError)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cls_incr_dataset(self):
        dataset_type = 'ClassIncDataset'
        data_file = os.path.join(self.assets_path, 'csvs/dss18.data.csv')
        anno_file = os.path.join(self.assets_path, 'csvs/dss18.anno.single_cls.csv')
        classes = ["Other", "Senior", "Kids", "Unknown"]
        self.data_cfg = dict(
            type=dataset_type,
            data_prefix=self.data_prefix,
            ann_file=anno_file,
            data_file=data_file,
            pipeline=self.pipeline,
            classes=classes
        )
        dataset = build_dataset(self.data_cfg)
        data_infos = dataset.data_infos
        self.assertTrue(isinstance(data_infos, list))
        self.assertTrue(isinstance(data_infos[0], dict))
        self.assertTrue(isinstance(data_infos[0]['gt_label'], np.ndarray))
        self.assertFalse(data_infos[0].get('soft_labels', False))
        self.assertFalse(data_infos[0].get('center', False))

        old_prob = np.random.rand(10, 4)
        center = np.random.rand(10, 10)
        for i, data_info in enumerate(data_infos):
            data_info['soft_label'] = old_prob[i]
            data_info['center'] = center[i]
        np.save('dummy_cls_inc.npy', data_infos)
        dataset_incr = build_dataset(self.data_cfg,
                                     default_args={'pre_stage_res': 'dummy_cls_inc.npy', 'dst_classes': classes})
        data_infos_incr = dataset_incr.data_infos
        self.assertEqual(len(data_infos_incr[0]['soft_label']), 4)
        self.assertEqual(len(data_infos_incr[0]['center']), 10)
        os.remove('dummy_cls_inc.npy')
