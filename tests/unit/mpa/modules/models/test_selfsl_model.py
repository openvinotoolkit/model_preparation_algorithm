import unittest
import pytest
import torch
# from mmcls.models import BACKBONES
from mpa.selfsl.builder import build_trainer
from mpa.modules.models.selfsl_model import SelfSL

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSL(unittest.TestCase):
    def setUp(self):
        self.model_cfg = dict(
            type='SelfSL',
            down_task='classification',
            pretrained='torchvision://resnet50',
            backbone=dict(
                type='ResNet',
                depth=50,
            ),
            neck=dict(
                type='MLP',
                in_channels=2048,
                hid_channels=256,
                out_channels=128,
                with_avg_pool=True
            ),
            head=dict(
                type='LatentPredictHead',
                loss='MSE',
                size_average=True,
                predictor=dict(
                    type='MLP',
                    in_channels=128,
                    hid_channels=256,
                    out_channels=128,
                    with_avg_pool=False
                )
            )
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build(self):
        trainer = build_trainer(self.model_cfg)
        self.assertTrue(isinstance(trainer, SelfSL))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_forward(self):
        trainer = build_trainer(self.model_cfg)
        dummy_image = torch.rand(4, 3, 112, 112)
        loss = trainer.train_step(dict(img1=dummy_image, img2=dummy_image), None)
        self.assertIsInstance(loss, dict)
        self.assertEqual(len(loss), 3)
        self.assertIn('loss', loss)
        self.assertEqual(len(loss['loss'].shape), 0)
