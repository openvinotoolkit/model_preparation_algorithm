# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from mmcv.runner import HOOKS, Hook, LrUpdaterHook
from mmcv.runner.hooks.ema import EMAHook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmdet.core.evaluation.eval_hooks import EvalHook
from mpa.utils.logger import get_logger
from mpa.modules.hooks.early_stopping_hook import EarlyStoppingHook

logger = get_logger()


@HOOKS.register_module()
class AdaptiveTrainingHook(Hook):
    """Adaptive Training Scheduling Hook

    Depending on the size of the dataset, adaptively update the validation interval and related values.
    Additionally, the momentum of EMA is also specified as adaptive.

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
    """

    def __init__(
        self,
        adaptive_ema_momentum=True,
        adaptive_validation_interval=True,
        max_interval=5,
        ema_interval=1,
        base_lr_patience=5,
        min_lr_patience=2,
        base_es_patience=10,
        min_es_patience=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.adaptive_ema_momentum = adaptive_ema_momentum
        self.adaptive_validation_interval = adaptive_validation_interval
        self.max_interval = max_interval
        self.ema_interval = ema_interval
        self.base_lr_patience = base_lr_patience
        self.min_lr_patience = min_lr_patience
        self.base_es_patience = base_es_patience
        self.min_es_patience = min_es_patience
        self.initialized = False
        self.enabled = False

    def before_train_epoch(self, runner):
        if not self.initialized:
            iter_per_epoch = len(runner.data_loader)
            adaptive_interval = self.get_adaptive_interval(iter_per_epoch)
            for hook in runner.hooks:
                if isinstance(hook, EMAHook) and self.adaptive_ema_momentum:
                    hook.momentum = self.get_adaptive_ema_momentum(
                        hook.momentum, iter_per_epoch
                    )
                    hook.interval = self.ema_interval
                if self.adaptive_validation_interval:
                    if isinstance(hook, EvalHook):
                        hook.interval = adaptive_interval
                        logger.info(f"Update Validation Interval: {adaptive_interval}")
                    elif isinstance(hook, LrUpdaterHook):
                        hook.interval = adaptive_interval
                        hook.patience = max(
                            math.ceil((self.base_lr_patience / adaptive_interval)),
                            self.min_lr_patience,
                        )
                        logger.info(f"Update Lr patience: {hook.patience}")
                    elif isinstance(hook, EarlyStoppingHook):
                        hook.start = adaptive_interval
                        hook.interval = adaptive_interval
                        hook.patience = max(
                            math.ceil((self.base_es_patience / adaptive_interval)),
                            self.min_es_patience,
                        )
                        logger.info(f"Update Early-Stop patience: {hook.patience}")
                    elif isinstance(hook, CheckpointHook):
                        hook.interval = adaptive_interval
            self.initialized = True

    def get_adaptive_ema_momentum(self, current_momentum, iter_per_epoch):
        epoch_decay = 1 - current_momentum
        iter_decay = math.pow(epoch_decay, self.ema_interval / iter_per_epoch)
        logger.info(f"Update EMA momentum: {1 - iter_decay}")
        return 1 - iter_decay

    def get_adaptive_interval(self, iter_per_epoch):
        decay = -0.025
        adaptive_interval = max(
            round(math.exp(decay * iter_per_epoch) * self.max_interval), 1
        )
        return adaptive_interval
