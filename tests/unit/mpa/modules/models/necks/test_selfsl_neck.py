import unittest
import pytest
import torch
from mpa.selfsl.builder import build_neck
from mpa.modules.models.necks.selfsl_neck import MLP, PPM

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSLNeck(unittest.TestCase):
    def setUp(self):
        self.mlp_neck_cfg = dict(
            type='MLP',
            in_channels=2048,
            hid_channels=256,
            out_channels=128,
            norm_cfg=dict(type='BN1d'),
            use_conv=False,
            with_avg_pool=True
        )
        self.ppm_neck_cfg = dict(
            type='PPM',
            sharpness=2
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_neck_MLP(self):
        neck = build_neck(self.mlp_neck_cfg)
        self.assertIsInstance(neck, MLP)

        input = torch.rand([3, 2048, 7, 7])
        out = neck(input)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, self.mlp_neck_cfg['out_channels']))

        with pytest.raises(TypeError):
            out = neck(input.numpy())

        self.mlp_neck_cfg['with_avg_pool'] = False
        neck = build_neck(self.mlp_neck_cfg)
        with pytest.raises(RuntimeError):
            out = neck(input)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_neck_PPM(self):
        neck = build_neck(self.ppm_neck_cfg)
        self.assertIsInstance(neck, PPM)

        input = torch.rand([3, 256, 7, 7])
        out = neck(input)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, input.shape)

        with pytest.raises(TypeError):
            out = neck(input.numpy())
