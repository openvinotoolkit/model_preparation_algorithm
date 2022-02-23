import unittest
import logging
import torch
from mmcv.runner import BaseRunner
from mpa.modules.hooks.selfsl_hook import SelfSLHook
import pytest

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


class Model(torch.nn.Module):
    def __init__(self, momentum):
        super().__init__()
        self.num_updated = 0
        self.base_momentum = momentum
        self.momentum = momentum

    def train_step(self):
        pass

    def momentum_update(self):
        self.num_updated += 1


class Runner(BaseRunner):
    def __init__(self, model, max_iters=None, max_epochs=None, iters_per_epoch=3):
        super(Runner, self).__init__(model, max_iters=max_iters, max_epochs=max_epochs, logger=logging.getLogger())
        self.iters_per_epoch = iters_per_epoch

    def train(self):
        pass

    def val(self):
        pass

    def run(self, *args, **kwargs):
        pass

    def save_checkpoint(self, *args, **kwargs):
        pass

    def run_epoch(self):
        self.call_hook('before_train_epoch')
        for i in range(self.iters_per_epoch):
            self._inner_iter = i
            self.run_iter()
        self._epoch += 1
        self.call_hook('after_train_epoch')

    def run_iter(self):
        self.call_hook('before_train_iter')
        self._iter += 1
        self.call_hook('after_train_iter')


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSLHook(unittest.TestCase):

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build_hook(self):
        import mmcv
        from mmcv.runner.hooks import HOOKS
        custom_hook_config = dict(type='SelfSLHook')
        hook = mmcv.build_from_cfg(custom_hook_config, HOOKS)
        self.assertIsInstance(hook, SelfSLHook)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_hook_by_iter(self):
        base_momentum = 0.9
        model = Model(base_momentum)
        runner = Runner(model, max_iters=4)

        custom_hook_config = dict(type='SelfSLHook')
        runner.register_hook_from_cfg(custom_hook_config)
        self.assertIsInstance(runner.hooks[0], SelfSLHook)

        runner.run_iter()
        self.assertEqual(model.momentum, base_momentum)

        for _ in range(int(runner.max_iters/2)):
            runner.run_iter()

        # 1 - ( 1- base_momentum) * (cos(pi * 1/2) + 1) / 2
        self.assertEqual(model.momentum, (1 + base_momentum)/2)

        for _ in range(int(runner.max_iters/2)):
            runner.run_iter()
        self.assertEqual(model.momentum, 1.0)
        self.assertEqual(model.num_updated, runner.max_iters + 1)

        # both momentum and weight should be updated
        old_momentum = model.momentum
        runner.run_iter()
        self.assertNotEqual(model.momentum, old_momentum)
        self.assertEqual(model.num_updated, runner.max_iters + 2)

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_hook_by_epoch(self):
        base_momentum = 0.9
        model = Model(base_momentum)
        runner = Runner(model, max_epochs=4)

        custom_hook_config = dict(type='SelfSLHook', by_epoch=True)
        runner.register_hook_from_cfg(custom_hook_config)
        self.assertIsInstance(runner.hooks[0], SelfSLHook)

        runner.run_epoch()
        self.assertEqual(model.momentum, base_momentum)

        for _ in range(int(runner.max_epochs/2)):
            runner.run_epoch()

        self.assertEqual(model.momentum, (1 + base_momentum)/2)

        for _ in range(int(runner.max_epochs/2)):
            runner.run_epoch()
        self.assertEqual(model.momentum, 1.0)
        self.assertEqual(model.num_updated, (runner.max_epochs + 1) * runner.iters_per_epoch)

        # momentum should NOT be updated while weight is updated
        old_momentum = model.momentum
        runner.run_iter()
        self.assertEqual(model.momentum, old_momentum)
        self.assertEqual(model.num_updated, (runner.max_epochs + 1) * runner.iters_per_epoch + 1)
