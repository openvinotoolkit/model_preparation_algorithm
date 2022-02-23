import unittest
import pytest
import torch

from mpa.modules.models.heads.multi_classifier_head import MultiClsHead
from mmcls.models.builder import build_head

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestMultiClsHead(unittest.TestCase):
    def setUp(self):
        self.in_channels = 500
        self.head_cfg = dict(
            type='MultiClsHead',
            in_channels=self.in_channels,
            tasks=dict(
                Gender=["Male", "Female", "Unknown"],
                Backpack=['Yes', 'No']
            )
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build(self):
        head = build_head(self.head_cfg)
        self.assertTrue(isinstance(head, MultiClsHead))
        with self.assertRaises(TypeError):
            self.head_cfg['tasks'] = 1
            head = build_head(self.head_cfg)
        with self.assertRaises(ValueError):
            self.head_cfg['tasks'] = dict()
            head = build_head(self.head_cfg)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_forward(self):
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(3, self.in_channels)
        dummy_gt = torch.tensor([[0, 0], [1, 1], [2, 0]])
        loss = head.forward_train(dummy_feature, dummy_gt)
        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 2)
        self.assertEqual(len(loss['loss'].shape), 0)
        self.assertTrue(isinstance(loss['accuracy'], dict))
        self.assertEqual(len(loss['accuracy']), 2)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_simple_test(self):
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(3, self.in_channels)
        features = head.simple_test(dummy_feature)
        self.assertTrue(isinstance(features, dict))
        self.assertEqual(len(features), 2)
        self.assertEqual(len(features['Gender'][0]), 3)
        self.assertEqual(len(features['Backpack'][0]), 2)
