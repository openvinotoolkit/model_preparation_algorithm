import unittest
import logging
import torch
from mmcls.models.builder import build_classifier
from mmcv.runner import BaseRunner, build_optimizer
from mpa.modules.hooks.model_ema_v2_hook import ModelEmaV2Hook, ModelEmaV2
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

    def run_epoch(self):
        self.call_hook("before_train_epoch")
        for i in range(self.iters_per_epoch):
            self._inner_iter = i
            self.run_iter()
        self._epoch += 1
        self.call_hook("after_train_epoch")

    def run_iter(self):
        self.call_hook("before_train_iter")
        self._iter += 1
        self.call_hook("after_train_iter")

    def save_checkpoint(self, *args, **kwargs):
        pass


@pytest.mark.components(MPAComponent.MPA)
class TestModelEMAV2Hook(unittest.TestCase):
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_model_ema_v2_hook(self):
        """
        Verifies that ModelEmaV2Hook builds
        """
        import mmcv
        from mmcv.runner.hooks import HOOKS

        custom_hook_config = dict(type="ModelEmaV2Hook")
        hook = mmcv.build_from_cfg(custom_hook_config, HOOKS)
        self.assertIsInstance(hook, ModelEmaV2Hook)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_model_ema_v2(self):
        """
        Verifies that ModelEmaV2Hook configs work
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
        runner = Runner(model, optimizer, max_epochs=3)

        custom_hook_config = dict(type="ModelEmaV2Hook", start_epoch=1)
        runner.register_hook_from_cfg(custom_hook_config)
        self.assertIsInstance(runner.hooks[0], ModelEmaV2Hook)

        runner.run()
        self.assertIsInstance(runner.ema_model, ModelEmaV2)

        for _ in range(2):
            runner.run_epoch()
