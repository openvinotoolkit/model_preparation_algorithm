# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook
from mmcv.runner import EpochBasedRunner

from mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class CancelInterfaceHook(Hook):
    def __init__(self, stop_flag=None, interval=5):
        self.runner = None
        self.interval = interval
        self.stop_flag = stop_flag
        
    def before_train_epoch(self, runner):
        if self.stop_flag:
            self.cancel()

    def cancel(self):
        logger.info('CancelInterfaceHook.cancel() is called.')
        if self.runner is None:
            logger.warning('runner is not configured yet. ignored this request.')
            return

        if self.runner.should_stop:
            logger.warning('cancel already requested.')
            return

        if isinstance(self.runner, EpochBasedRunner):
            epoch = self.runner.epoch
            self.runner._max_epochs = epoch  # Force runner to stop by pretending it has reached it's max_epoch
        self.runner.should_stop = True  # Set this flag to true to stop the current training epoch
        logger.info('requested stopping to the runner')

    def before_run(self, runner):
        self.runner = runner
