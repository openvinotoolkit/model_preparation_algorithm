import unittest
import os
import shutil
import pytest
import time
from unittest.mock import patch, MagicMock, Mock

from mmcv.utils import Config, ConfigDict

from mpa.det.stage import DetectionStage

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestDetectionStage(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure(self):
        recipe_cfg = Config(dict(
            task_adapt=dict(
                type='',
                op='REPLACE',
            ),
            hyperparams=dict(
            ),
            work_dir='./logs',
            log_level='INFO',
        ))
        model_cfg = Config(dict(
            model=dict(
                type='',
            ),
        ))
        model_cfg2 = Config(dict(
            model=dict(
                type='',
                task='other',
            ),
        ))
        data_cfg = Config(dict(
            data=dict(
            ),
        ))
        kwargs = dict(
            pretrained='pretrained.pth',
        )

        with patch('mpa.det.stage.Stage.__init__'):
            stage = DetectionStage()
            stage.cfg = recipe_cfg
            stage._init_logger()
        stage.configure_model = MagicMock()
        stage.configure_data = MagicMock()
        stage.configure_task = MagicMock()
        stage.configure_regularization = MagicMock()
        stage.configure_hyperparams = MagicMock()
        stage.configure_hook = MagicMock()
        cfg = stage.configure(model_cfg, 'model.pth', data_cfg, training=True, **kwargs)
        stage.configure_model.assert_called()
        stage.configure_data.assert_called()
        stage.configure_task.assert_called()
        stage.configure_regularization.assert_called()
        stage.configure_hyperparams.assert_called()
        stage.configure_hook.assert_called()
        with self.assertRaises(ValueError):
            cfg = stage.configure(model_cfg2, 'model.pth', data_cfg, training=True, **kwargs)
        with self.assertRaises(ValueError):
            cfg = stage.configure(Config(dict(a=1)), 'model.pth', data_cfg, training=True, **kwargs)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_model(self):
        cfg = Config(dict(
            model=dict(
                type='SSD',
                super_type='UnbiasedTeacher',
                backbone=dict(
                    type='mobilenetv2_w1',
                ),
                bbox_head=dict(
                    type='SSDHead',
                ),
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'):
            DetectionStage().configure_model(cfg, training=False)
        self.assertEqual(cfg.model.type, 'UnbiasedTeacher')
        self.assertEqual(cfg.model.arch_type, 'SSD')
        self.assertEqual(cfg.model.bbox_head.type, 'PseudoSSDHead')

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_get_model_classes(self):
        cfg = Config(dict(
            load_from='somewhere',
            model=dict(
                classes=['person', 'car',],
            ),
        ))
        with patch('mpa.det.stage.torch.load', return_value={}):
            classes = DetectionStage.get_model_classes(cfg)
        self.assertEqual(classes, ['person', 'car',])

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_data(self):
        cfg = Config(dict(
            data=dict(
                train=dict(
                    type='OrgType',
                    super_type='SuperType',
                    pipeline=[dict(type='TrainPipe'),],
                ),
                unlabeled=dict(
                    img_file='unlabeled.json',
                ),
                samples_per_gpu=2,
                workers_per_gpu=2,
            ),
            seed=1234,
            custom_hooks=[],
        ))
        with patch('mpa.det.stage.Stage.__init__'), patch('mpa.det.stage.Stage.configure_data') as func:
            DetectionStage().configure_data(cfg, training=True)
        func.assert_called()
        self.assertEqual(cfg.data.train.type, 'SuperType')
        self.assertEqual(cfg.data.train.org_type, 'OrgType')
        self.assertEqual(cfg.data.unlabeled.pipeline, [dict(type='TrainPipe'),])
        self.assertEqual(len(cfg.custom_hooks), 1)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_task(self):
        cfg = Config(dict(
            task_adapt=dict(
                type='',
                op='REPLACE',
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'):
            stage = DetectionStage()

        stage.configure_task_classes = MagicMock(
            return_value=(['person'], ['car'], ['car'])
        )
        stage.configure_task_data_pipeline = MagicMock()
        stage.configure_task_eval_dataset = MagicMock()
        stage.configure_task_adapt_hook = MagicMock()
        stage.configure_anchor = MagicMock()
        stage.configure_task_cls_incr = MagicMock()
        stage.configure_task(cfg, training=True)
        stage.configure_task_classes.assert_called()
        stage.configure_task_data_pipeline.assert_not_called()
        stage.configure_task_eval_dataset.assert_called()
        stage.configure_task_adapt_hook.assert_called()
        stage.configure_anchor.assert_not_called()
        stage.configure_task_cls_incr.assert_called()

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_task_classes(self):
        cfg = Config(dict(
            task_adapt=dict(
                final=[],
            ),
            model=dict(
                task_adapt=dict(
                    src_classes=[],
                    dst_classes=[],
                ),
                bbox_head=dict(
                    num_classes=0,
                ),
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'), patch('mpa.det.stage.DetectionStage.get_model_classes', return_value=['person']), patch('mpa.det.stage.DetectionStage.get_data_classes', return_value=['car']):
            stage = DetectionStage()
            org_model_classes, model_classes, data_classes = \
                stage.configure_task_classes(cfg, '', 'REPLACE')
            self.assertEqual(model_classes, ['car'])
            self.assertEqual(cfg.model.bbox_head.num_classes, 1)
            org_model_classes, model_classes, data_classes = \
                stage.configure_task_classes(cfg, '', 'MERGE')
            self.assertEqual(model_classes, ['person', 'car'])
            self.assertEqual(cfg.model.bbox_head.num_classes, 2)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_task_data_pipeline(self):
        cfg = Config(dict(
            data=dict(
                train=dict(
                    pipeline=[
                        dict(type='LoadAnnotations'),
                        dict(type='Other'),
                    ],
                ),
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'), patch('mpa.det.stage.DetectionStage.get_model_classes', return_value=['person']), patch('mpa.det.stage.DetectionStage.get_data_classes', return_value=['car']):
            stage = DetectionStage()
            stage.configure_task_data_pipeline(cfg, ['person', 'car'], ['car'])
            self.assertEqual(len(cfg.data.train.pipeline), 3)
            stage.configure_task_data_pipeline(cfg, ['person', 'car'], ['car'])
            self.assertEqual(len(cfg.data.train.pipeline), 3)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_task_eval_dataset(self):
        cfg = Config(dict(
            data=dict(
                val=dict(
                    type='CustomDataset',
                    org_type='',
                    model_classes=[],
                ),
                test=dict(
                    type='CustomDataset',
                    org_type='',
                    model_classes=[],
                ),
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'):
            stage = DetectionStage()
            stage.configure_task_eval_dataset(cfg, ['person', 'car'])
            self.assertEqual(cfg.data.val.type, 'TaskAdaptEvalDataset')
            self.assertEqual(cfg.data.val.org_type, 'CustomDataset')
            self.assertEqual(cfg.data.val.model_classes, ['person', 'car'])
            self.assertEqual(cfg.data.test.type, 'TaskAdaptEvalDataset')
            self.assertEqual(cfg.data.test.org_type, 'CustomDataset')
            self.assertEqual(cfg.data.test.model_classes, ['person', 'car'])

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_task_adapt_hook(self):
        cfg = Config(dict(
            model=dict(type='VFNet'),
            task_adapt=dict(),
            custom_hooks=[],
        ))
        with patch('mpa.det.stage.Stage.__init__'):
            stage = DetectionStage()
            stage.configure_task_adapt_hook(cfg, ['person'], ['person', 'car'])
            self.assertEqual(len(cfg.custom_hooks), 1)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_task_cls_incr(self):
        cfg = Config(dict(
            task_adapt=dict(
                type='mpa',
            ),
            model=dict(
                type='SSD',
                bbox_head=dict(
                    type='SSDHead',
                ),
            ),
            data=dict(
                train=dict(
                    type='CocoDataset',
                ),
            ),
            custom_hooks=[],
        ))
        with patch('mpa.det.stage.Stage.__init__'), patch('mpa.det.stage.DetectionStage.get_img_ids_for_incr', return_value={}):
            stage = DetectionStage()
            stage.configure_task_cls_incr(cfg, 'mpa', ['person'], ['person', 'car'])
            self.assertEqual(len(cfg.custom_hooks), 2)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_regularization(self):
        cfg = Config(dict(
            model=dict(
                l2sp_weight=1.0,
                pretrained='pretrained.pth',
            ),
            load_from='load_from.pth',
            optimizer=dict(
                weight_decay=0.1,
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'):
            stage = DetectionStage()
            stage.configure_regularization(cfg)
            self.assertEqual(cfg.model.l2sp_ckpt, 'load_from.pth')
            self.assertEqual(cfg.optimizer.weight_decay, 0.0)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure_hparams(self):
        cfg = Config(dict(
            data=dict(
                samples_per_gpu=2,
            ),
            optimizer=dict(
                lr=0.1,
            ),
        ))
        hparams = dict(hyperparams=dict(
            bs=8,
            lr=0.02,
        ))
        with patch('mpa.det.stage.Stage.__init__'):
            DetectionStage().configure_hyperparams(cfg, training=True, **hparams)
        self.assertEqual(cfg.data.samples_per_gpu, 8)
        self.assertEqual(cfg.optimizer.lr, 0.02)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_update_or_add_custom_hook(self):
        cfg = Config(dict(
            custom_hooks=[],
        ))
        hook1 = dict(
            type='SomeHook',
            param=1,
        )
        hook2 = dict(
            type='SomeHook',
            param=2,
        )
        DetectionStage.update_or_add_custom_hook(cfg, hook1)
        self.assertEqual(len(cfg.custom_hooks), 1)
        DetectionStage.update_or_add_custom_hook(cfg, hook2)
        self.assertEqual(len(cfg.custom_hooks), 1)
        self.assertEqual(cfg.custom_hooks, [hook2])
