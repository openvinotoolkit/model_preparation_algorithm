import unittest
import pytest

import torch

from mpa.modules.models.losses.cross_entropy_loss import CrossEntropyLossWithIgnore, WeightedCrossEntropyLoss
from mmcls.models.builder import build_loss

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestCrossEntropyLosses(unittest.TestCase):
    def setUp(self):
        self.loss_cfg_compare = {"type": "CrossEntropyLoss", "loss_weight": 1.0}
        self.compute_loss_compare = build_loss(self.loss_cfg_compare)

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_ce_w_ignore_loss(self):
        self.loss_cfg = {"type": "CrossEntropyLossWithIgnore", "loss_weight": 1.0, "ignore_index": -1}
        compute_loss = build_loss(self.loss_cfg)
        self.assertTrue(isinstance(compute_loss, CrossEntropyLossWithIgnore))
        dummy_score = torch.rand(4, 3)
        dummy_gt = torch.tensor([0, 1, 2, -1])
        loss = compute_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)

        dummy_score = dummy_score[0:3]
        dummy_gt = dummy_gt[0:3]
        loss_compare = self.compute_loss_compare(dummy_score, dummy_gt)
        self.assertAlmostEqual(loss.cpu().numpy() * 4 / 3, loss_compare.cpu().numpy(), 3)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_weighted_ce_loss(self):
        self.loss_cfg = {"type": "WeightedCrossEntropyLoss", "loss_weight": 1.0, "class_weight": [0.0, 1, 1]}
        compute_loss = build_loss(self.loss_cfg)
        self.assertTrue(isinstance(compute_loss, WeightedCrossEntropyLoss))
        dummy_score = torch.rand(4, 3)
        dummy_gt = torch.tensor([1, 1, 2, 0])
        if torch.cuda.is_available():
            dummy_score = dummy_score.cuda()
            dummy_gt = dummy_gt.cuda()
        loss = compute_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)

        dummy_score = dummy_score[0:3]
        dummy_gt = dummy_gt[0:3]
        loss_compare = self.compute_loss_compare(dummy_score, dummy_gt)
        self.assertAlmostEqual(loss.cpu().numpy() * 4 / 3, loss_compare.cpu().numpy(), 3)
