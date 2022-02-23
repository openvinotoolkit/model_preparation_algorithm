import unittest
import pytest
from unittest.mock import Mock, MagicMock

from mpa.workflow import Workflow
from mpa.modules.hooks.workflow_hook import WorkflowHook
from mpa.stage import Stage

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_init(self):
        with pytest.raises(Exception) as e:
            Workflow('test', None)
        self.assertEqual(e.type, ValueError)

        with pytest.raises(Exception) as e:
            Workflow([], None)
        self.assertEqual(e.type, ValueError)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_call_hooks(self):
        wf_hook0 = Mock(spec=WorkflowHook)
        wf_hook0.before_workflow = MagicMock()
        wf_hook1 = Mock(spec=WorkflowHook)
        wf_hook1.after_workflow = MagicMock()
        wf_hook2 = Mock(spec=WorkflowHook)
        wf_hook2.before_stage = MagicMock()
        wf_hook3 = Mock(spec=WorkflowHook)
        wf_hook3.after_stage = MagicMock()

        stage0 = Mock(spec=Stage)
        stage0.name = 'fake0'
        stage0.run = MagicMock(return_value=dict(key='xyz'))
        wf = Workflow([stage0], [wf_hook0, wf_hook1, wf_hook2, wf_hook3])
        wf.run()

        wf_hook0.before_workflow.assert_called()
        wf_hook1.after_workflow.assert_called()
        wf_hook2.before_stage.assert_called()
        wf_hook3.after_stage.assert_called()

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_run(self):
        stage0 = Mock(spec=Stage)
        stage0.name = 'fake0'
        stage0.run = MagicMock(return_value=dict(key='xyz'))
        stage1 = Mock(spec=Stage)
        stage1.name = 'fake1'
        stage1.run = MagicMock(return_value=dict())
        stage1.input = dict(
            stage0_output=dict(
                stage_name='fake0',
                output_key='key'
            )
        )
        stage_invalid = Mock(spec=Stage)
        stage_invalid.name = 'invalid'
        stage_invalid.run = MagicMock(return_value='run-results1')
        stage_invalid.input = dict(
            # missing required attribute definition
            stage0_output=dict(
                stage_name='fake0'
            )
        )

        wf = Workflow([stage0, stage_invalid], None)
        with pytest.raises(Exception) as e:
            wf.run()
        self.assertEqual(e.type, ValueError)

        wf = Workflow([stage0, stage1], None)
        wf.run()
