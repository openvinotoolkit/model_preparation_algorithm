import unittest
import pytest

import torch

from mpa.modules.models.losses.distillation_losses import LwfLoss
from mmcls.models.builder import build_loss

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestDistillationLoss(unittest.TestCase):
    def setUp(self):
        self.loss_cfg = dict(
            type='LwfLoss',
            loss_weight=1.0
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_loss(self):
        compute_loss = build_loss(self.loss_cfg)
        self.assertTrue(isinstance(compute_loss, LwfLoss))
        dummy_score = torch.rand(4, 3)
        dummy_gt = torch.rand(4, 3)
        loss = compute_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)
