import os.path as osp
import sys
import runpy
import pytest

from tests.constants.setup_values import IntegrationTestsSetup

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestIntegrationSegmentation:
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "segmentor",
        [
            'litehrnet18.custom.yaml'
        ]
    )
    def test_recipe_seg_class_incr(finalizer_for_intg, segmentor):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/segmentors', segmentor)
        model_ckpt = osp.join(IntegrationTestsSetup.assets_path, 'model_cfg/ckpt/seg_cityscapes_car_imgnet.pth')
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/cityscapes_seg_class_incr.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'seg_class_incr.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --model_ckpt {model_ckpt} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_train/latest.pth')

        assert osp.exists(output_model)