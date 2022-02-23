import os.path as osp
import sys
import runpy
import pytest

from tests.constants.setup_values import IntegrationTestsSetup

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestIntegrationClassification:
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "classifier",
        [
            'mnet_v2.yaml',
            'ote_mnet_v3_small.yaml',
            'ote_mnet_v3_large.yaml',
            'ote_mnet_v3_large_075.yaml',
            'ote_effnet_b0.yaml',
            'resnet18.yaml',
            'ote_mnet_v3_large.nonlinear_head.yaml'
        ]
    )
    def test_recipe_cls_ft_with_classifiers(finalizer_for_intg, classifier):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/classifiers', classifier)
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/cifar10split_224_bs16.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls.yaml')
        argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} --model_cfg {model_cfg} ' \
               '--recipe_hparams runner.max_epochs=2 ' \
               f'--output_path {IntegrationTestsSetup.output_path}'
        sys.argv = argv.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_finetune/latest.pth')
        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "dataset",
        [
            'cifar10split_224_bs16.py',
            'svhnsplit_bs16.py'
        ]
    )
    def test_recipe_cls_ft_with_datasets(finalizer_for_intg, dataset):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/classifiers/mnet_v2.yaml')
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/', dataset)
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls.yaml')
        argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} --model_cfg {model_cfg} ' \
               '--recipe_hparams runner.max_epochs=2 ' \
               f'--output_path {IntegrationTestsSetup.output_path}'
        sys.argv = argv.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_finetune/latest.pth')
        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "params",
        [
            {'model': 'mobilenet_v2.yaml', 'ckpt': 'cifar10_5cls_mnet_v2.pth'},
            {'model': 'ote_mobilenet_v3_large_075.yaml', 'ckpt': 'cifar10_5cls_ote-mnet_v3_large075.pth'}
        ]
    )
    def test_recipe_cls_class_incr_with_models(finalizer_for_intg, params):
        model_cfg = osp.join(IntegrationTestsSetup.model_path, 'classification/', params['model'])
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/cifar10_224_cls_inc.py')
        model_ckpt = osp.join(IntegrationTestsSetup.assets_path, 'model_cfg/ckpt/', params['ckpt'])
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls_class_incr.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --model_ckpt {model_ckpt} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_cls-inc/latest.pth')

        assert osp.exists(output_model)

        # Efficient Mode
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --model_ckpt {model_ckpt} \
                     --recipe_hparams runner.max_epochs=2 task_adapt.efficient_mode=True \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_cls-inc/latest.pth')

        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "backbone",
        [
            'mobilenet_v2.yaml',
            'ote_mobilenet_v3_small.yaml',
            'ote_mobilenet_v3_large.yaml',
            'ote_mobilenet_v3_large_075.yaml',
            'ote_efficientnet_b0.yaml'
        ]
    )
    def test_recipe_cls_semisl_with_backbones(finalizer_for_intg, backbone):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/backbones', backbone)
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/stl10_224-bs8-40-280.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls_semisl.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} --model_cfg {model_cfg} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_semisl-fixmatch/latest.pth')

        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "dataset",
        [
            'cifar10_224-bs8-40-280.py',
            'stl10_224-bs8-40-280.py',
            'fmnist_224-bs8-40-280.py',
            'svhn_224-bs8-40-280.py',
        ]
    )
    def test_recipe_cls_semisl_with_datasets(finalizer_for_intg, dataset):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/backbones/ote_mobilenet_v3_small.yaml')
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg', dataset)
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls_semisl.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} --model_cfg {model_cfg} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage00_semisl-fixmatch/latest.pth')

        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "backbone",
        [
            'mobilenet_v2.yaml',
            'ote_mobilenet_v3_small.yaml',
            'ote_mobilenet_v3_large.yaml',
            'ote_mobilenet_v3_large_075.yaml',
            'ote_efficientnet_b0.yaml'
        ]
    )
    @pytest.mark.skip(reason='deprecated')
    def test_recipe_cls_selfsl_with_backbones(finalizer_for_intg, backbone):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/backbones', backbone)
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/stl10split_224.py')
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls_selfsl.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} --model_cfg {model_cfg} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage01_cls_from_byol/latest.pth')

        assert osp.exists(output_model)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.component
    @pytest.mark.parametrize(
        "params",
        [
            {'backbone': 'backbones/mobilenet_v2.yaml', 'ckpt': 'dss18_gen_bp_lh_mnetv2.pth'},
        ]
    )
    def test_recipe_cls_taskinc_with_backbones(finalizer_for_intg, params):
        model_cfg = osp.join(IntegrationTestsSetup.sample_path, 'cfgs/models/', params['backbone'])
        data_cfg = osp.join(IntegrationTestsSetup.assets_path, 'data_cfg/dss18_cls_task_incr.py')
        model_ckpt = osp.join(IntegrationTestsSetup.assets_path, 'model_cfg/ckpt/', params['ckpt'])
        recipe = osp.join(IntegrationTestsSetup.recipes_path, 'cls_task_incr.yaml')
        sys.argv = f'tools/cli.py {recipe} --data_cfg {data_cfg} \
                     --model_cfg {model_cfg} \
                     --model_ckpt {model_ckpt} \
                     --recipe_hparams runner.max_epochs=2 \
                     --output_path {IntegrationTestsSetup.output_path}'.split()

        runpy.run_module('tools.cli', run_name='__main__')
        output_model = osp.join(IntegrationTestsSetup.output_path, 'latest/stage01_task-inc/latest.pth')

        assert osp.exists(output_model)
