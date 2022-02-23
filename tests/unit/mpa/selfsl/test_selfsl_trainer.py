"""
import unittest
import pytest
from unittest.mock import patch, Mock
from mmcv import Config
from mpa.selfsl.trainer import SelfSLTrainer

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSLStageTrainer(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_selfsl_stage_trainer(self):
        selfsl_cfg = 'recipes/stages/classification/train_selfsl.yaml'
        with patch('mpa.selfsl.trainer.SelfSLTrainer'):
            stage = SelfSLTrainer(config=selfsl_cfg, name='selfsl_training', mode='train')
            model_cfg = Config.fromfile('models/classification/resnet50.yaml')
            data_cfg = Mock(data=Mock(unlabeled=None))
            out = stage.run(model_cfg=model_cfg, data_cfg=data_cfg, model_ckpt=None)
            self.assertIsInstance(out, dict)
            self.assertIn('pretrained', out)
        self.assertIsInstance(stage, SelfSLTrainer)
"""

import copy
import os
import os.path as osp
import shutil
import unittest
from unittest.mock import patch, Mock
import torch
import pytest

from mmcv import ConfigDict

from mpa.selfsl.trainer import SelfSLTrainer

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


# @pytest.mark.components(MPAComponent.MPA)
def check_dict(cfg, pairs: dict, parents: list, check: dict):
    for k, v in cfg.items():
        parents.append(k)
        full_k = '.'.join(parents)
        if (k in pairs.keys() and full_k not in pairs.keys() and v != pairs[k]) \
                or (full_k in pairs.keys() and v != pairs[full_k]):
            check[full_k] = v
        elif isinstance(v, dict):
            check_dict(v, pairs, parents, check)
        elif isinstance(v, (list, tuple)):
            for i in range(len(v)):
                _v = v[i]
                parents.append(str(i))
                if isinstance(_v, dict):
                    check_dict(_v, pairs, parents, check)
                parents.pop()
        parents.pop()


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSLTrainer(unittest.TestCase):
    def setUp(self):
        self.assets_path = 'tests/assets'
        self.output_path = 'outputs_selfsl_trainer'
        self.model_cfg = dict(
            model=dict(
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    norm_eval=True
                )
            )
        )
        self.dataset_cfg = dict(
            data=dict(
                unlabeled=dict(
                    type='TVDatasetSplit',
                    base='CIFAR10',
                    pipeline=[],
                    train=True,
                    download=True,
                    num_images=100
                )
            )
        )
        os.makedirs(self.output_path, exist_ok=True)

    def tearDown(self):
        if osp.exists(self.output_path):
            shutil.rmtree(self.output_path)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_trainer(self):
        cfg_dict = dict(type='test-selfsl')
        with pytest.raises(KeyError):
            trainer = SelfSLTrainer(config=cfg_dict, name='stage', mode='train')
            self.assertIsInstance(trainer, SelfSLTrainer)

        cfg_file = 'recipes/stages/classification/selfsl.yaml'
        trainer = SelfSLTrainer(config=cfg_file, name='stage', mode='train')
        self.assertIsInstance(trainer, SelfSLTrainer)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_configure(self):
        cfg_file = 'recipes/stages/classification/selfsl.yaml'
        backbone_cfg = ConfigDict(copy.deepcopy(self.model_cfg['model']['backbone']))
        dataset_cfg = ConfigDict(copy.deepcopy(self.dataset_cfg))

        trainer = SelfSLTrainer(config=cfg_file, name='stage', mode='train')
        trainer.configure(backbone_cfg, dataset_cfg)
        self.assertTrue(bool(trainer.cfg.model.pretrained))
        pairs = {'norm_eval': False}
        check = {}
        check_dict(trainer.cfg, pairs, [], check)
        self.assertFalse(check, f'Expected {pairs}')

        trainer._configure_model(backbone_cfg, 'random')
        self.assertFalse(bool(trainer.cfg.model.pretrained))

        trainer._configure_model(backbone_cfg, 'model_ckpt.pth')
        self.assertEqual(trainer.cfg.load_from, 'model_ckpt.pth')

        backbone_cfg['type'] = 'OTEEfficientNet'
        converted_key = osp.join(self.output_path, 'model_ckpt.converted.pth')
        feat = Mock(spec=torch.Tensor, shape=(2, 1024, 7, 7))
        backbone = Mock(return_value=feat)
        with patch('mpa.selfsl.trainer.convert_keys', Mock(return_value=converted_key)), \
                patch('mpa.selfsl.trainer.build_backbone', Mock(return_value=backbone)):
            trainer._configure_model(backbone_cfg, 'model_ckpt.pth')
        self.assertEqual(trainer.cfg.load_from, osp.join(self.output_path, 'model_ckpt.converted.pth'))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_run(self):
        cfg_file = 'recipes/stages/classification/selfsl.yaml'
        model_cfg = ConfigDict(copy.deepcopy(self.model_cfg))
        data_cfg = ConfigDict(copy.deepcopy(self.dataset_cfg))

        trainer = SelfSLTrainer(config=cfg_file, name='stage', mode='train')

        # set for short tests
        trainer.cfg.data.pipeline_options.RandomResizedCrop.size = 32
        trainer.cfg.work_dir = self.output_path
        trainer.cfg.checkpoint_config.interval = 1
        trainer.cfg.runner.max_iters = 1
        trainer.cfg.runner.max_epochs = None

        trainer.run(model_cfg=model_cfg, data_cfg=data_cfg, model_ckpt=None)
        self.assertTrue(osp.exists(self.output_path))
        self.assertTrue(osp.exists(osp.join(self.output_path, 'backbone.pth')))
