import unittest
import pytest
import torch

from mpa.modules.models.heads.cls_incremental_head import ClsIncrHead
from mmcls.models.builder import build_head

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClsIncClsHead(unittest.TestCase):
    def setUp(self):
        self.in_channels = 1280
        self.num_classes = 10
        self.num_old_classes = 7
        self.head_cfg = dict(
            type='ClsIncrHead',
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            num_old_classes=self.num_old_classes,
            distillation_loss=dict(type='LwfLoss', T=2.0, loss_weight=1.0),
            ranking_loss=dict(type="TripletLoss", margin=0.3, dist_metric="cosine")
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build(self):
        head = build_head(self.head_cfg)
        self.assertTrue(isinstance(head, ClsIncrHead))
        with self.assertRaises(TypeError):
            self.head_cfg['num_old_classes'] = [1]
            head = build_head(self.head_cfg)
        with self.assertRaises(ValueError):
            self.head_cfg['num_old_classes'] = 0
            head = build_head(self.head_cfg)
        with self.assertRaises(TypeError):
            self.head_cfg['num_classes'] = [1]
            head = build_head(self.head_cfg)
        with self.assertRaises(ValueError):
            self.head_cfg['num_classes'] = 0
            head = build_head(self.head_cfg)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_forward(self):
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(16, self.in_channels)
        dummy_gt = torch.randint(self.num_classes, (16,))
        dummy_soft_labels = torch.rand(16, self.num_old_classes)
        dummy_center = torch.rand(16, self.in_channels)

        loss = head.forward_train(dummy_feature, dummy_gt, dummy_soft_labels, dummy_center)

        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 5)
        self.assertEqual(len(loss['cls_loss'].shape), 0)
        self.assertEqual(len(loss['dist_loss'].shape), 0)
        self.assertEqual(len(loss['center_loss'].shape), 0)
        self.assertEqual(len(loss['ranking_loss'].shape), 0)
        self.assertTrue(isinstance(loss['accuracy'], dict))
        self.assertEqual(len(loss['accuracy']), 1)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_simple_test(self):
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(3, self.in_channels)
        features = head.simple_test(dummy_feature)
        self.assertEqual(len(features), 3)
        self.assertEqual(len(features[0]), self.num_classes)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_prob_extract(self):
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(3, self.in_channels)
        features = head.extract_prob(dummy_feature)
        self.assertEqual(len(features), 3)
        self.assertEqual(len(features[0]), self.num_old_classes)
