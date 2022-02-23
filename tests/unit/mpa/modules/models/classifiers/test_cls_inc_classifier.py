import unittest
import pytest

import torch

from mpa.modules.models.classifiers.cls_incremental_classifier import ClsIncrementalClassifier
from mmcls.models.builder import build_classifier

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestClsIncClassifier(unittest.TestCase):
    def setUp(self):
        self.in_channels = 1280
        self.num_classes = 10
        self.num_old_classes = 3
        self.model_cfg = dict(
            type='ClsIncrementalClassifier',
            backbone=dict(
                type='MobileNetV2',
                widen_factor=1.0,
            ),
            neck=dict(
                type='GlobalAveragePooling',
            ),
            head=dict(
                type='ClsIncrHead',
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                num_old_classes=self.num_old_classes,
                distillation_loss=dict(type='LwfLoss', T=2.0, loss_weight=1.0),
                ranking_loss=dict(type="TripletLoss", margin=0.3, dist_metric="cosine")
            )
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build(self):
        classifier = build_classifier(self.model_cfg)
        self.assertTrue(isinstance(classifier, ClsIncrementalClassifier))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_forward(self):
        classifier = build_classifier(self.model_cfg)
        dummy_image = torch.rand(16, 3, 224, 224)
        dummy_gt = torch.randint(self.num_classes, (16,))
        dummy_soft_labels = torch.rand(16, self.num_old_classes)
        dummy_center = torch.rand(16, self.in_channels)

        loss = classifier.forward_train(dummy_image, dummy_gt, soft_label=dummy_soft_labels, center=dummy_center)
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
    def test_extract_prob(self):
        classifier = build_classifier(self.model_cfg)
        dummy_image = torch.rand(16, 3, 224, 224)
        probs, feats = classifier.extract_prob(dummy_image)
        self.assertEqual(len(probs), 16)
        self.assertEqual(len(probs[0]), self.num_old_classes)
        self.assertEqual(len(feats[0]), self.in_channels)
