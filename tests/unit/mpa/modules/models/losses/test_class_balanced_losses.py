import unittest
import pytest

import torch

from mpa.modules.models.losses.class_balanced_losses import SoftmaxFocalLoss, SoftmaxPolarityLoss
from mmcls.models.builder import build_loss

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClassBalancedLosses(unittest.TestCase):
    def setUp(self):
        self.focal_loss_cfg = dict(
            type='SoftmaxFocalLoss',
            loss_weight=1.0
        )
        self.polarity_loss_cfg = dict(
            type='SoftmaxPolarityLoss',
            loss_weight=1.0
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_loss(self):
        compute_focal_loss = build_loss(self.focal_loss_cfg)
        self.assertTrue(isinstance(compute_focal_loss, SoftmaxFocalLoss))
        dummy_score = torch.rand(4, 3)
        dummy_gt = torch.tensor([0, 1, 2, 1])
        loss = compute_focal_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)

        compute_polarity_loss = build_loss(self.polarity_loss_cfg)
        self.assertTrue(isinstance(compute_polarity_loss, SoftmaxPolarityLoss))
        loss = compute_focal_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)

        # ignore index
        focal_loss_cfg = dict(
            type='SoftmaxFocalLoss',
            loss_weight=1.0,
            ignore_index=2
        )
        compute_focal_loss = build_loss(focal_loss_cfg)
        dummy_score = torch.rand(4, 3)
        dummy_gt = torch.tensor([0, 1, 2, 1])
        loss = compute_focal_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)

        polarity_loss_cfg = dict(
            type='SoftmaxPolarityLoss',
            loss_weight=1.0,
            ignore_index=2
        )
        compute_polarity_loss = build_loss(polarity_loss_cfg)
        dummy_score = torch.rand(4, 3)
        dummy_gt = torch.tensor([0, 1, 2, 1])
        loss = compute_polarity_loss(dummy_score, dummy_gt)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(len(loss.shape), 0)
