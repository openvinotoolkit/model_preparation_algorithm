import unittest
import pytest

import torch

from mpa.modules.models.losses.triplet_loss import TripletLoss
from mmcls.models.builder import build_loss

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestTripletLoss(unittest.TestCase):
    def setUp(self):
        self.loss_cfg = {"type": "TripletLoss", "margin": 0.3, "dist_metric": "cosine"}

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_loss(self):
        compute_loss = build_loss(self.loss_cfg)
        self.assertTrue(isinstance(compute_loss, TripletLoss))
        dummy_features = torch.rand(8, 128)
        dummy_gts = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loss = compute_loss(dummy_features, dummy_gts)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)
