import unittest
import pytest

import torch

from mpa.modules.models.classifiers.task_incremental_classifier import TaskIncrementalLwF
from mmcls.models.builder import build_classifier

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestTaskIncClassifier(unittest.TestCase):
    def setUp(self):
        self.model_cfg = dict(
            type='TaskIncrementalLwF',
            backbone=dict(
                type='MobileNetV2',
                widen_factor=1.0,
            ),
            neck=dict(
                type='GlobalAveragePooling',
            ),
            head=dict(
                type='TaskIncLwfHead',
                in_channels=1280,
                tasks=dict(
                    Age=["Other", "Senior", "Kids", "Unknown"]
                ),
                old_tasks=dict(
                    Gender=["Male", "Female", "Unknown"],
                    Backpack=['Yes', 'No']
                )
            )
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build(self):
        classifier = build_classifier(self.model_cfg)
        self.assertTrue(isinstance(classifier, TaskIncrementalLwF))

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_forward(self):
        classifier = build_classifier(self.model_cfg)
        dummy_image = torch.rand(3, 3, 224, 224)
        dummy_gt = torch.tensor([[0], [1], [2]])
        dummy_soft_labels = dict(
            Gender=torch.rand(3, 3),
            Backpack=torch.rand(3, 2)
        )
        loss = classifier.forward_train(dummy_image, dummy_gt, soft_label=dummy_soft_labels)
        self.assertTrue(isinstance(loss, dict))
        self.assertEqual(len(loss), 3)
        self.assertEqual(len(loss['new_loss'].shape), 0)
        self.assertEqual(len(loss['old_loss'].shape), 0)
        self.assertTrue(isinstance(loss['accuracy'], dict))
        self.assertEqual(len(loss['accuracy']), 1)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_extract_prob(self):
        model_cfg = dict(
            type='TaskIncrementalLwF',
            backbone=dict(
                type='MobileNetV2',
                widen_factor=1.0,
            ),
            neck=dict(
                type='GlobalAveragePooling',
            ),
            head=dict(
                type='TaskIncLwfHead',
                in_channels=1280,
                tasks=dict(
                    Gender=["Male", "Female", "Unknown"],
                    Backpack=['Yes', 'No']
                )
            )
        )
        classifier = build_classifier(model_cfg)
        dummy_image = torch.rand(3, 3, 224, 224)
        probs, _ = classifier.extract_prob(dummy_image)
        self.assertTrue(isinstance(probs, dict))
        self.assertEqual(len(probs), 2)
        self.assertEqual(len(probs['Gender'][0]), 3)
        self.assertEqual(len(probs['Backpack'][0]), 2)
