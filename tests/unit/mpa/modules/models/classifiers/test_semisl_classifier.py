import unittest
import pytest

import torch

from mmcls.models.builder import build_classifier
from mpa.modules.models.classifiers.semisl_classifier import SemiSLClassifier

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSemiSLClassifier(unittest.TestCase):
    def setUp(self):
        """
        Semi-SL Classifier settings with MV2
        """
        self.in_channels = 576
        self.num_classes = 3
        self.model_cfg = dict(
            type="SemiSLClassifier",
            backbone=dict(
                type="OTEMobileNetV3",
                mode="small",
                width_mult=1.0
            ),
            neck=dict(
                type="GlobalAveragePooling",
            ),
            head=dict(
                type="SemiSLClsHead",
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            ),
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_classifier_build(self):
        """
        Verifies that classifier build works
        """
        classifier = build_classifier(self.model_cfg)
        self.assertTrue(isinstance(classifier, SemiSLClassifier))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_classifier_forward(self):
        """
        Verifies that the forward_train function works
        """
        classifier = build_classifier(self.model_cfg)
        # Labeled images
        dummy_labeled_image = torch.rand(3, 3, 32, 32)
        # Labeled GT
        dummy_gt = torch.tensor([0, 1, 2])
        # Unlabeled images
        dummy_data = {
            "weak": {
                "img": torch.rand(9, 3, 32, 32),
                "gt_label": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
            },
            "strong": {"img": torch.rand(9, 3, 32, 32)},
        }

        loss = classifier.forward_train(
            imgs=dummy_labeled_image, gt_label=dummy_gt, extra_0=dummy_data
        )
        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 2)
        self.assertTrue(isinstance(loss["accuracy"], dict))
        self.assertEqual(len(loss["accuracy"]), 1)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_classifier_pretrained_init_weight(self):
        """
        Verifies that initialization weight of backbone, neck, head
        with pretrained weights
        """
        cfg = self.model_cfg
        cfg["pretrained"] = True
        classifier = build_classifier(cfg)
        # Labeled images
        dummy_labeled_image = torch.rand(3, 3, 32, 32)
        # Labeled GT
        dummy_gt = torch.tensor([0, 1, 2])
        # Unlabeled images
        dummy_data = {
            "weak": {
                "img": torch.rand(9, 3, 32, 32),
                "gt_label": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
            },
            "strong": {"img": torch.rand(9, 3, 32, 32)},
        }

        loss = classifier.forward_train(
            imgs=dummy_labeled_image, gt_label=dummy_gt, extra_0=dummy_data
        )
        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 2)
        self.assertTrue(isinstance(loss["accuracy"], dict))
        self.assertEqual(len(loss["accuracy"]), 1)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_classifier_extract_feat(self):
        """
        Verifies that classifier's extract features from the backbone + neck
        """
        classifier = build_classifier(self.model_cfg)
        dummy_image = torch.rand(3, 3, 32, 32)
        feats = classifier.extract_feat(dummy_image)
        self.assertTrue(isinstance(feats, torch.Tensor))
        self.assertEqual(len(feats), 3)

        model_cfg = dict(
            type="SemiSLClassifier",
            backbone=dict(
                type="MobileNetV2",
                widen_factor=1.0,
            ),
            head=dict(
                type="SemiSLClsHead",
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            ),
        )
        classifier = build_classifier(model_cfg)
        dummy_image = torch.rand(3, 3, 32, 32)
        feats = classifier.extract_feat(dummy_image)
        self.assertTrue(isinstance(feats, torch.Tensor))
        self.assertEqual(len(feats), 3)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_classifier_simple_test(self):
        """
        Verifies that classifier use simple_test function with small data
        """
        classifier = build_classifier(self.model_cfg)
        dummy_image = torch.rand(3, 3, 32, 32)
        features = classifier.simple_test(dummy_image)
        self.assertEqual(len(features), 3)
        self.assertEqual(len(features[0]), self.num_classes)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_classifier_value_error(self):
        """
        Verifies that occur ValueError in an incorrect data configuration
        (without gt_label or unlabeled data)
        """
        classifier = build_classifier(self.model_cfg)
        # Labeled images
        dummy_labeled_image = torch.rand(3, 3, 32, 32)
        # Labeled GT
        dummy_gt = torch.tensor([0, 1, 2])
        # Unlabeled images
        dummy_data = {
            "weak": {
                "img": torch.rand(9, 3, 32, 32),
                "gt_label": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
            },
            "strong": {"img": torch.rand(9, 3, 32, 32)},
        }

        with self.assertRaises(ValueError):
            classifier.forward_train(
                imgs=dummy_labeled_image, extra_0=dummy_data
            )
        with self.assertRaises(ValueError):
            classifier.forward_train(
                imgs=dummy_labeled_image, gt_label=dummy_gt
            )
