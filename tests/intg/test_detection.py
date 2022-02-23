import os.path as osp
import sys
import runpy
import pytest

from tests.constants.setup_values import IntegrationTestsSetup

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements
# from e2e.constants.framework_constants import FrameworkMessages


@pytest.mark.components(MPAComponent.MPA)
class TestIntegrationDetection:
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "detector",
        [
            # 'frcnn_r50.custom.yaml',
            'ssd_mv2w1.custom.yaml',
            'atss_mv2w1.custom.yaml',
            'vfnet_r50.custom.yaml',
        ]
    )
    def test_recipe_det_custom(finalizer_for_intg, detector):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/detectors', detector)
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/coco_nopipe_smallest_resize.person.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'det.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_train/latest.pth')

        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "detector",
        [
            # 'frcnn_r50.custom.yaml',
            'ssd_mv2w1.custom.yaml',
            'atss_mv2w1.custom.yaml',
            'vfnet_r50.custom.yaml',
        ]
    )
    def __test_recipe_det_class_incr(finalizer_for_intg, detector):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/detectors', detector)
        model_ckpt = osp.join(IntegrationTestsSetup.assets_path, 'model_cfg/ckpt/frcnn_rnet50_coco-10-person.zip')
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/coco_resize_smallest_car.repeat.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'det_class_incr.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --model_ckpt {model_ckpt} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_train/latest.pth')

        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "detector",
        [
            # 'frcnn_r50.custom.yaml',
            'ssd_mv2w1.custom.yaml',
            'atss_mv2w1.custom.yaml',
            'vfnet_r50.custom.yaml',
        ]
    )
    def test_recipe_det_semisl(finalizer_for_intg, detector):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/detectors', detector)
        model_ckpt = osp.join(IntegrationTestsSetup.assets_path, 'model_cfg/ckpt/frcnn_rnet50_coco-10-person.zip')
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/coco_nopipe_smallest_resize.person.semi.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'det_semisl.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --model_ckpt {model_ckpt} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_train/latest.pth')

        assert osp.exists(output_model)
