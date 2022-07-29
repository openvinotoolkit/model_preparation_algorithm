# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ResumeTrainHook(Hook):

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.args = kwargs

    def before_run(self, runner):
        runner.resume(
            checkpoint=self.checkpoint,
            resume_optimizer=True,
            **self.args
        )
        runner.logger.info('resume in ResumeTrainHook is done.')
