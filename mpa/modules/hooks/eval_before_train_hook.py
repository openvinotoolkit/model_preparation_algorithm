# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook
from mmdet.core.evaluation.eval_hooks import EvalHook as DetEvalHook
from mmcv.runner import EvalHook as MMCVEvalHook
from mpa.modules.hooks.eval_hook import CustomEvalHook as ClsEvalHook

@HOOKS.register_module()
class EvalBeforeTrainHook(Hook):
    """Hook to evaluate before training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._executed = False

    def before_train_epoch(self, runner):
        """Execute evaluation hook before training"""
        if not self._executed:
            for hook in runner.hooks:
                if self.check_eval_hook(hook):
                    if hook.by_epoch:
                        hook.after_train_epoch(runner)
                    else:
                        hook.after_train_iter(runner)
                    break
            self._executed = True

    @staticmethod
    def check_eval_hook(hook):
        """Check that hook is in charge of evaluation."""
        hook_class = type(hook)
        return (
            issubclass(hook_class, DetEvalHook)
            or issubclass(hook_class, MMCVEvalHook)  # check mmseg eval hooks
            or issubclass(hook_class, ClsEvalHook)
        )
