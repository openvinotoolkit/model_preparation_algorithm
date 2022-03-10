import unittest
import os
import shutil
import pytest
import time
from unittest.mock import patch, MagicMock, Mock

from mmcv.utils import Config, ConfigDict

from mpa.det.inferrer import DetectionInferrer

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
    def test_run(self):
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
        data_cfg = Config(dict(
            data=dict(
            ),
        ))
        kwargs = dict(
            pretrained='pretrained.pth',
            mode='infer',
        )

        with patch('mpa.det.stage.Stage.__init__'), patch('mpa.det.inferrer.np.save'):
            stage = DetectionInferrer()
            stage.cfg = recipe_cfg
            stage.mode = ['infer']
        super(Config, stage.cfg).__setattr__('dump', MagicMock())
        stage.configure = MagicMock(return_value=stage.cfg)
        stage.infer = MagicMock(return_value=dict(detections=[]))
        infer_result = stage.run(model_cfg, 'model.pth', data_cfg, **kwargs)
        stage.configure.assert_called()
        stage.cfg.dump.assert_called()
        stage.infer.assert_called()
        self.assertEqual(infer_result['output_file_path'], './logs/infer_result.npy')

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_infer(self):
        cfg = Config(dict(
            task_adapt=dict(
                type='',
                op='REPLACE',
                final=['person', 'car'],
            ),
            hyperparams=dict(
            ),
            work_dir='./logs',
            log_level='INFO',
            load_from='pretrained.pth',
            model=dict(
                type='',
                neck=[dict()],
            ),
            data=dict(
                samples_per_gpu=1,
                workers_per_gpu=2,
                test=dict(
                    ann_file='test.json',
                    img_prefix='imgs',
                    classes=['person', 'car'],
                ),
            ),
        ))
        with patch('mpa.det.stage.Stage.__init__'), \
                patch('mpa.det.inferrer.build_dataset') as build_dataset, \
                patch('mpa.det.inferrer.build_dataloader') as build_dataloader, \
                patch('mpa.det.inferrer.build_detector') as build_detector, \
                patch('mpa.det.inferrer.load_checkpoint') as load_checkpoint:
            stage = DetectionInferrer()
            stage.cfg = cfg
            stage._init_logger()
            infer_result = stage.infer(cfg)
            build_dataset.assert_called()
            build_dataloader.assert_called()
            build_detector.assert_called()
            load_checkpoint.assert_called()
