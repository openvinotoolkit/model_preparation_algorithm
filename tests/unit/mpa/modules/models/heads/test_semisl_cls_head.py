import unittest
import pytest
import torch

from mpa.modules.models.heads.semisl_cls_head import SemiSLClsHead
from mmcls.models.builder import build_head

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSemiSLClsHead(unittest.TestCase):
    def setUp(self):
        """
        Semi-SL for Classification Head Settings
        """
        self.in_channels = 1280
        self.num_classes = 10
        self.head_cfg = dict(
            type="SemiSLClsHead",
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_semisl_cls_head(self):
        """
        Verifies that SemiSLClsHead builds
        """
        head = build_head(self.head_cfg)
        self.assertTrue(isinstance(head, SemiSLClsHead))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_semisl_cls_head_type_error(self):
        """
        Verifies that SemiSLClsHead parameters check with TypeError
        """
        with self.assertRaises(TypeError):
            self.head_cfg["num_classes"] = [1]
            build_head(self.head_cfg)
        with self.assertRaises(TypeError):
            self.head_cfg["in_channels"] = [1]
            build_head(self.head_cfg)
        with self.assertRaises(TypeError):
            self.head_cfg["loss"] = [1]
            build_head(self.head_cfg)
        with self.assertRaises(TypeError):
            self.head_cfg["topk"] = [1]
            build_head(self.head_cfg)
        with self.assertRaises(TypeError):
            self.head_cfg["unlabeled_coef"] = [1]
            build_head(self.head_cfg)
        with self.assertRaises(TypeError):
            self.head_cfg["min_threshold"] = [1]
            build_head(self.head_cfg)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_semisl_cls_head_value_error(self):
        """
        Verifies that SemiSLClsHead parameters check with ValueError
        """
        with self.assertRaises(ValueError):
            self.head_cfg["num_classes"] = 0
            build_head(self.head_cfg)
        with self.assertRaises(ValueError):
            self.head_cfg["num_classes"] = -1
            build_head(self.head_cfg)
        with self.assertRaises(ValueError):
            self.head_cfg["in_channels"] = 0
            build_head(self.head_cfg)
        with self.assertRaises(ValueError):
            self.head_cfg["in_channels"] = -1
            build_head(self.head_cfg)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_forward(self):
        """
        Verifies that SemiSLClsHead forward function works
        """
        head = build_head(self.head_cfg)
        labeled_batch_size = 16
        unlabeled_batch_size = 64

        dummy_gt = torch.randint(self.num_classes, (labeled_batch_size,))
        dummy_data = {
            "labeled": torch.rand(labeled_batch_size, self.in_channels),
            "unlabeled_weak": torch.rand(unlabeled_batch_size, self.in_channels),
            "unlabeled_strong": torch.rand(unlabeled_batch_size, self.in_channels),
        }

        loss = head.forward_train(dummy_data, dummy_gt)

        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 2)
        self.assertTrue(isinstance(loss["accuracy"], dict))
        self.assertEqual(len(loss["accuracy"]), 2)

        # No Unlabeled Data
        dummy_feature = torch.rand(labeled_batch_size, self.in_channels)
        loss = head.forward_train(dummy_feature, dummy_gt)

        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 2)
        self.assertTrue(isinstance(loss["accuracy"], dict))
        self.assertEqual(len(loss["accuracy"]), 2)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_simple_test(self):
        """
        Verifies that SemiSLClsHead simple_test function works
        """
        head = build_head(self.head_cfg)
        dummy_feature = torch.rand(3, self.in_channels)
        features = head.simple_test(dummy_feature)
        self.assertEqual(len(features), 3)
        self.assertEqual(len(features[0]), self.num_classes)
