import unittest
import os
import shutil
import pytest
import time
from unittest.mock import patch, MagicMock, Mock

from mmcv.utils import Config, ConfigDict

from mpa.det.evaluator import DetectionEvaluator

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
            mode='eval',
        )

        with patch('mpa.det.stage.Stage.__init__'), patch('mpa.det.evaluator.json.dump'):
            stage = DetectionEvaluator()
            stage.cfg = recipe_cfg
            stage.mode = ['eval']
        super(Config, stage.cfg).__setattr__('dump', MagicMock())
        stage.configure = MagicMock(return_value=stage.cfg)
        stage.infer = MagicMock(return_value=dict(detections=[]))
        stage.dataset = MagicMock()
        stage.dataset.evaluate = MagicMock(return_value=dict(bbox_mAP_50=0.5))
        eval_result = stage.run(model_cfg, 'model.pth', data_cfg, **kwargs)
        stage.configure.assert_called()
        stage.cfg.dump.assert_called()
        stage.infer.assert_called()
        stage.dataset.evaluate.assert_called()
        self.assertEqual(eval_result['mAP'], 0.5)

