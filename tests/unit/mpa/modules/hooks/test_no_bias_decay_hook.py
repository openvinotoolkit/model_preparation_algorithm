import unittest
import logging
import torch
from mmcls.models.builder import build_classifier
from mmcv.runner import BaseRunner, build_optimizer
from mpa.modules.hooks.no_bias_decay_hook import NoBiasDecayHook
import pytest

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()


class Runner(BaseRunner):
    def __init__(
        self, model, optimizer, max_iters=None, max_epochs=None, iters_per_epoch=3
    ):
        super(Runner, self).__init__(
            model,
            max_iters=max_iters,
            max_epochs=max_epochs,
            logger=logging.getLogger(),
        )
        self.iters_per_epoch = iters_per_epoch
        self.model = model
        self.optimizer = optimizer

    def train(self):
        pass

    def val(self):
        pass

    def run(self, *args, **kwargs):
        self.call_hook("before_run")

    def save_checkpoint(self, *args, **kwargs):
        pass


@pytest.mark.components(MPAComponent.MPA)
class TestNoBiasDecayHook(unittest.TestCase):
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_no_bias_decay_hook(self):
        """
        Verifies that NoBiasDecayHook builds
        """
        import mmcv
        from mmcv.runner.hooks import HOOKS

        custom_hook_config = dict(type="NoBiasDecayHook")
        hook = mmcv.build_from_cfg(custom_hook_config, HOOKS)
        self.assertIsInstance(hook, NoBiasDecayHook)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_no_bias_decay(self):
        """
        Verifies that NoBiasDecayHook config & works
        """
        model_cfg = dict(
            type="ImageClassifier",
            backbone=dict(
                type="MobileNetV2",
                widen_factor=1.0,
            ),
            neck=dict(
                type="GlobalAveragePooling",
            ),
            head=dict(
                type="LinearClsHead",
                in_channels=1280,
                num_classes=3,
            ),
        )
        optimizer_config = dict(type="SGD", lr=0.03, momentum=0.9)
        model = build_classifier(model_cfg)
        optimizer = build_optimizer(model, optimizer_config)
        runner = Runner(model, optimizer, max_epochs=5)

        custom_hook_config = dict(type="NoBiasDecayHook")
        runner.register_hook_from_cfg(custom_hook_config)
        self.assertIsInstance(runner.hooks[0], NoBiasDecayHook)

        runner.run()
        self.assertEqual(len(runner.optimizer.param_groups), 3)
