import unittest
import logging
import torch
import math
import pytest

from mmcls.models.builder import build_classifier
from mmcv.runner import BaseRunner, build_optimizer
from mmcv.parallel import is_module_wrapper

from mpa.modules.hooks.semisl_cls_hook import SemiSLClsHook

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
        pass

    def run_epoch(self):
        self.call_hook("before_train_epoch")
        for i in range(self.iters_per_epoch):
            self._inner_iter = i
            self.run_iter()
        self._epoch += 1
        self.call_hook("after_epoch")

    def run_iter(self):
        self.call_hook("before_train_iter")
        self._iter += 1
        self.call_hook("after_train_iter")

    def save_checkpoint(self, *args, **kwargs):
        pass


@pytest.mark.components(MPAComponent.MPA)
class TestSemiSLClsHook(unittest.TestCase):
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_semi_sl_cls_hook(self):
        """
        Verifies that SemiSLClsHook builds
        """
        import mmcv
        from mmcv.runner.hooks import HOOKS

        custom_hook_config = dict(type="SemiSLClsHook")
        hook = mmcv.build_from_cfg(custom_hook_config, HOOKS)
        self.assertIsInstance(hook, SemiSLClsHook)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_semi_sl_cls_hook(self):
        """
        Verifies that SemiSLClsHook's warm-up loss working & update log_buffer
        """
        model_cfg = dict(
            type="SemiSLClassifier",
            backbone=dict(
                type="MobileNetV2",
                widen_factor=1.0,
            ),
            neck=dict(
                type="GlobalAveragePooling",
            ),
            head=dict(
                type="SemiSLClsHead",
                in_channels=1280,
                num_classes=3,
            ),
        )
        optimizer_config = dict(type="SGD", lr=0.03, momentum=0.9)
        model = build_classifier(model_cfg)
        optimizer = build_optimizer(model, optimizer_config)
        runner = Runner(model, optimizer, max_iters=10)

        custom_hook_config = dict(type="SemiSLClsHook")
        runner.register_hook_from_cfg(custom_hook_config)
        self.assertIsInstance(runner.hooks[0], SemiSLClsHook)

        if is_module_wrapper(runner.model):
            head = runner.model.module.head
        else:
            head = runner.model.head

        mu = (
            lambda x: 0.50
            - math.cos(min(math.pi, (2 * math.pi * x) / runner.max_iters)) / 2
        )

        for i in range(runner.max_iters):
            runner.run_iter()
            self.assertEqual(head.unlabeled_coef, mu(i))

        runner.run_epoch()
        self.assertTrue(isinstance(runner.log_buffer.output, dict))
        self.assertEqual(runner.log_buffer.output['unlabeled_loss'], 1.0)
        self.assertIn('pseudo_label', runner.log_buffer.output)
