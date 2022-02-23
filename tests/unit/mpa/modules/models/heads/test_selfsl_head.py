import unittest
import torch

from mpa.selfsl.builder import build_head
from mpa.modules.models.heads.selfsl_head import LatentPredictHead
import pytest

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSLHead(unittest.TestCase):
    def setUp(self):
        self.mlp_head_cfg = dict(
            type='LatentPredictHead',
            loss='MSE',
            predictor=dict(
                type='MLP',
                in_channels=128,
                hid_channels=256,
                out_channels=128,
                with_avg_pool=False
            )
        )
        self.ppm_head_cfg = dict(
            type='LatentPredictHead',
            loss='PPC',
            predictor=dict(
                type='PPM',
                sharpness=2,
            )
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_head_MLP(self):
        head = build_head(self.mlp_head_cfg)
        self.assertIsInstance(head, LatentPredictHead)
        online = torch.rand([3, 128])
        target = torch.rand([3, 128])
        loss = head(online, target)
        self.assertIsInstance(loss, dict)
        self.assertTrue(loss['loss'] >= 0 and loss['loss'] <= 4)

        self.mlp_head_cfg['size_average'] = False
        head = build_head(self.mlp_head_cfg)
        loss = head(online, target)

        self.assertIsInstance(loss, dict)
        self.assertTrue(loss['loss'] >= 0 and loss['loss'] <= 4*online.size(0))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_head_PPM(self):
        head = build_head(self.ppm_head_cfg)
        self.assertIsInstance(head, LatentPredictHead)
        online = torch.rand([3, 256, 7, 7])
        target = torch.rand([3, 256, 7, 7])
        coord_q = torch.rand([3, 4])
        coord_k = torch.rand([3, 4])
        loss = head(online, target, coord_q, coord_k)
        self.assertIsInstance(loss, dict)
