import unittest
import pytest
from unittest.mock import patch, Mock

from mmcv import Config
from mpa.builder import build
from mpa.workflow import Workflow
from mpa.stage import Stage

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestBuilder(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_workflow_config(self):
        config_file = 'path/not/exits/file.yaml'
        with pytest.raises(Exception) as e:
            build(config_file)
        self.assertEqual(e.type, ValueError)

        config = Config(
            dict(
                workflow_hooks=[
                    dict(
                        type='SampleLoggingHook',
                        log_level='WARNING'
                    )
                ],
                stages=[
                    dict(),
                    dict()
                ]
            )
        )
        mock_stage = Mock(spec=Stage)
        mock_stage.name = 'fake-stage'

        mock_build_stage = Mock(return_value=mock_stage)
        with patch('mpa.builder.__build_stage', mock_build_stage):
            wf = build(config)
        self.assertTrue(isinstance(wf, Workflow))

        config_file = 'recipes/cls.yaml'
        wf = build(config_file)
        self.assertTrue(isinstance(wf, Workflow))
