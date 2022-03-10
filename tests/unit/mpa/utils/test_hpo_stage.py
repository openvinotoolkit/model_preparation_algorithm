import unittest
import pytest

from mmcv.utils.config import ConfigDict

from mpa.utils.hpo_stage import HpoRunner
from mpa.utils.config_utils import MPAConfig

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
class TestHpoStage(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

    def tearDown(self):
        pass

    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_asha(self):
        cfg = None
        with pytest.raises(Exception) as e:
            hpo_runner = HpoRunner(config=cfg, mode='train', name='hpo')
        self.assertEqual(e.type, ValueError)

        hpo_cfg = {'hyperparams': [{'name': 'lr', 'range': [0.0001, 0.1], 'type': 'loguniform'},
                                   {'name': 'bs', 'range': [4, 32, 2], 'type': 'qloguniform'}],
                   'max_iterations': 1,
                   'metric': 'accuracy_top-1',
                   'num_trials': 1,
                   'search_alg': 'asha',
                   'subset_size': 100,
                   'trainer': {'config': 'recipes_old/classification/domain_adapt/finetune/train.yaml',
                               'type': 'ClsTrainer'},
                   }
        common_cfg = {'log_level': 'DEBUG'}
        hpo_runner = HpoRunner(hpo=ConfigDict(hpo_cfg),
                               common_cfg=ConfigDict(common_cfg),
                               mode='train',
                               name='hpo')
        self.assertIsInstance(hpo_runner, HpoRunner, 'cannot create a HpoRunner instance with Config object')
        self.assertEqual(hpo_runner.min_iterations, 1)
        self.assertEqual(hpo_runner.max_iterations, 1)
        self.assertEqual(hpo_runner.reduction_factor, 4)

        print('============================================')
        print(hpo_runner.cfg)
        print('============================================')
        print("common_cfg", hpo_runner.cfg['common_cfg'])
        print('============================================')

        cls_model_cfg = MPAConfig.fromfile('models/classification/mobilenet_v2.yaml')
        cifar_data_cfg = MPAConfig.fromfile(f'{self.assets_path}/data_cfg/cifar10split_224_bs16.py')

        best_cfg = hpo_runner.run(mode='train',
                                  model_cfg=cls_model_cfg,
                                  model_ckpt=None,
                                  data_cfg=cifar_data_cfg)

        self.assertIn('hyperparams', best_cfg)
        self.assertIn('lr', best_cfg['hyperparams'])
        self.assertIn('bs', best_cfg['hyperparams'])
        self.assertGreaterEqual(best_cfg['hyperparams']['lr'], 0.0001)
        self.assertLessEqual(best_cfg['hyperparams']['lr'], 0.1)
        self.assertGreaterEqual(best_cfg['hyperparams']['bs'], 4)
        self.assertLessEqual(best_cfg['hyperparams']['bs'], 32)
