import unittest
import os
import shutil
import pytest
import time
from unittest.mock import patch, MagicMock, Mock

from mmcv.utils import Config

from mpa.stage import Stage

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestStage(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_init(self):
        # config
        cfg = 'path/is/not/existed.yaml'
        with pytest.raises(Exception) as e:
            stage = Stage('stage', 'test-mode', cfg)
        self.assertEqual(e.type, ValueError)

        cfg = Mock(spec=Config, total_epochs=10)
        cfg.checkpoint_config = Mock(interval=1)
        cfg.runner = Mock(max_epochs=1)
        cfg.pop = Mock(return_value=1)
        cfg.get = Mock(return_value='')
        stage = Stage('stage', 'test-mode', cfg)
        self.assertIsInstance(stage, Stage, 'cannot create a Stage instance with Config object')

        cfg = dict(
            checkpoint_config=dict(interval=1),
            runner=dict(type='fast-runner', max_epochs=1)
        )
        stage = Stage('stage', 'test-mode', cfg)
        self.assertIsInstance(stage, Stage, 'cannot create a Stage instance with dict')

        cfg = 'recipes/stages/classification/train.yaml'
        stage = Stage('stage', 'test-mode', cfg)
        self.assertIsInstance(stage, Stage, 'cannot create a Stage instance with the config file')

        # common_cfg
        common_cfg = []
        with pytest.raises(Exception) as e:
            stage = Stage('stage', 'test-mode', cfg, common_cfg=common_cfg)
        self.assertEqual(e.type, TypeError)

        common_cfg = dict(runner=dict(type='new-runner', max_epochs=1))
        stage = Stage('stage', 'test-mode', cfg, common_cfg=common_cfg)
        self.assertIsInstance(stage, Stage, 'cannot create a Stage instance with the common_cfg')
        self.assertEqual(common_cfg['runner'], stage.cfg['runner'])

        # kwargs
        logger_info = MagicMock()
        with patch('mpa.utils.logger.info', logger_info):
            stage = Stage('stage', 'test-mode', cfg, something_new='new-value')
        logger_info.assert_called()
        self.assertEqual('new-value', stage.cfg['something_new'])

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_init_logger(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        time_strftime = MagicMock(return_value=timestamp)
        cfg = dict(
            log_level='DEBUG'
        )
        common_cfg = dict(
            output_path='logs/unittests'
        )
        stage = Stage('stage', 'test-mode', cfg, common_cfg)

        with patch('time.strftime', time_strftime):
            stage._init_logger()

        log_file = os.path.join(common_cfg['output_path'], f'stage00_stage/{timestamp}.log')
        self.assertTrue(os.path.exists(log_file))

        # cleanup output dir
        shutil.rmtree(common_cfg['output_path'], ignore_errors=False)
