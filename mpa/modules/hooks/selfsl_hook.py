from math import cos, pi
from mmcv.parallel import is_module_wrapper
from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS


@HOOKS.register_module()
class SelfSLHook(Hook):
    """Hook for SelfSL.

    This hook includes momentum adjustment in SelfSL following:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: total steps.

    Args:
        end_momentum (float): The final momentum coefficient
            for the target network. Default: 1.
    """

    def __init__(self, end_momentum=1., update_interval=1, by_epoch=False, **kwargs):
        self.by_epoch = by_epoch
        self.end_momentum = end_momentum
        self.update_interval = update_interval

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if not hasattr(model, 'momentum'):
            raise AttributeError("The SelfSL model must have attribute \"momentum\".")
        if not hasattr(model, 'base_momentum'):
            raise AttributeError("The SelfSL model must have attribute \"base_momentum\".")

        if self.every_n_epochs(runner, self.update_interval):
            cur_epoch = runner.epoch
            max_epoch = runner.max_epochs
            base_m = model.base_momentum
            m = self.end_momentum - (self.end_momentum - base_m) * (
                cos(pi * cur_epoch / float(max_epoch)) + 1) / 2
            model.momentum = m

    def before_train_iter(self, runner):
        if self.by_epoch:
            return

        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if not hasattr(model, 'momentum'):
            raise AttributeError("The SelfSL model must have attribute \"momentum\".")
        if not hasattr(model, 'base_momentum'):
            raise AttributeError("The SelfSL model must have attribute \"base_momentum\".")

        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            base_m = model.base_momentum
            m = self.end_momentum - (self.end_momentum - base_m) * (
                cos(pi * cur_iter / float(max_iter)) + 1) / 2
            model.momentum = m

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()
