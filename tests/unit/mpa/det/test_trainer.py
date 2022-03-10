import unittest
import os
import shutil
import pytest
import time
from unittest.mock import patch, MagicMock, Mock

from mmcv.utils import Config, ConfigDict

from mpa.det.trainer import DetectionTrainer

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
        cfg = Config(dict(
            task_adapt=dict(
                type='',
                op='REPLACE',
                final=['person', 'car'],
            ),
            hparams=dict(
                #adaptive_anchor=True,
            ),
            work_dir='./logs',
            log_level='INFO',
            gpu_ids=0,
            seed=1234,
            checkpoint_config=dict(
            ),
            model=dict(
                type='',
            ),
            data=dict(
                train=dict(
                ),
            ),
            dist_params=dict(
                linear_scale_lr=True,
            ),
            optimizer=dict(
                lr=0.001,
            ),
        ))
        kwargs = dict(
            pretrained='pretrained.pth',
            mode='train',
        )

        with patch('mpa.det.stage.Stage.__init__'), \
                patch('mpa.det.trainer.collect_env', return_value={}) as collect_env, \
                patch('mpa.det.trainer.build_dataset') as build_dataset, \
                patch('mpa.det.trainer.extract_anchor_ratio') as extract_anchor_ratio, \
                patch('mpa.det.trainer.DetectionTrainer.train_worker') as train_worker, \
                patch('mpa.det.trainer.mp.spawn') as spawn, \
                patch('mpa.det.trainer.glob.glob', return_value=['best.pth']) as glob:
            stage = DetectionTrainer()
            stage.cfg = cfg
            stage.mode = ['train']
            super(Config, stage.cfg).__setattr__('dump', MagicMock())
            stage.configure = MagicMock(return_value=stage.cfg)
            stage.configure_anchor = MagicMock()
            train_result = stage.run(Config(), 'model.pth', Config(), **kwargs)
            collect_env.assert_called()
            build_dataset.assert_called()
            extract_anchor_ratio.assert_not_called()
            train_worker.assert_called()
            spawn.assert_not_called()
            glob.assert_called()
            stage.configure.assert_called()
            stage.configure_anchor.assert_not_called()
            stage.cfg.dump.assert_called()
            self.assertEqual(train_result['final_ckpt'], 'best.pth')
            cfg.gpu_ids = (0, 1)
            train_result = stage.run(Config(), 'model.pth', Config(), **kwargs)
            spawn.assert_called()

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_train_worker(self):
        cfg = Config(dict(
            gpu_ids=(0,1),
            model=dict(
            ),
            dist_params=dict(
            ),
        ))

        with patch('mpa.det.trainer.torch.cuda.set_device') as set_device, \
                patch('mpa.det.trainer.dist.init_process_group') as init_process_group, \
                patch('mpa.det.trainer.dist.get_world_size') as get_world_size, \
                patch('mpa.det.trainer.dist.get_rank') as get_rank, \
                patch('mpa.det.trainer.build_detector') as build_detector, \
                patch('mpa.det.trainer.train_detector') as train_detector:
            DetectionTrainer.train_worker((0,1), ['person', 'car'], [], cfg, distributed=True)
            set_device.assert_called()
            init_process_group.assert_called()
            get_world_size.assert_called()
            get_rank.assert_called()
            build_detector.assert_called()
            train_detector.assert_called()
            DetectionTrainer.train_worker((0,1), ['person', 'car'], [], cfg, distributed=False)
            set_device.assert_called_once()
            init_process_group.assert_called_once()
            get_world_size.assert_called_once()
            get_rank.assert_called_once()
