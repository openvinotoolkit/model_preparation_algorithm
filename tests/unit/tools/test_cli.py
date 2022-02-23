import unittest
import pytest
import sys
import runpy

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestCli(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'
        self.recipes_path = 'recipes'
        self.argv_tmp = sys.argv
        self.output_path = 'outputs/tests/cli'

    def tearDown(self):
        sys.argv = self.argv_tmp

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_cli_missing_required(self):
        sys.argv = 'tools/cli.py'.split()
        with pytest.raises(SystemExit) as e:
            runpy.run_module('tools.cli', run_name='__main__')
        self.assertEqual(e.type, SystemExit)
        self.assertTrue(e.value.code, 2)
