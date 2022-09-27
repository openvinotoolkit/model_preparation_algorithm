# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import BaseRunner
from ote_sdk.utils.argument_checks import check_input_parameters_type


@HOOKS.register_module()
class SwitchPipelineHook(Hook):
    """
    Switch pipeline with specific interation.
    There are two types of switching pipeline:
        1. switching pipeline for each specific iteration
        2. applying the second pipeline intermittently with interval
    
    They can be controlled with `iteration` or `interval`.

    :param iteration: (NotImplemented) For 1. If `iteration` > 0, the pipeline will be switched for each specific `iteration`.
    :param interval: For 2. If `interval` > 1, the second pipeline will be used once per set `interval`. 
    """

    @check_input_parameters_type()
    def __init__(self, iteration: int = 0, interval: int = 1):
        # TODO : implement the case of `iteration` > 0
        if iteration > 0:
            raise NotImplementedError(
                'Switching pipeline for each specific iteration is not implemented.')

        assert not (iteration != 0 and interval != 1), \
            'Both methods cannot be use at the same time.'

        self.iteration = iteration
        self.interval = interval
        self.cnt = 0

    @check_input_parameters_type()
    def get_dataset(self, runner: BaseRunner):
        if hasattr(runner.data_loader.dataset, 'dataset'):
            # for RepeatDataset
            dataset = runner.data_loader.dataset.dataset
        else:
            dataset = runner.data_loader.dataset

        return dataset

    @check_input_parameters_type()
    def before_train_epoch(self, runner: BaseRunner):
        if self.interval > 1 and self.cnt == self.interval-1:
            # start supcon training
            # TODO : not using list index for stability
            dataset = self.get_dataset(runner)
            dataset.pipeline.transforms[2].is_supervised = False

        else:
            # TODO : not using list index for stability
            dataset = self.get_dataset(runner)
            dataset.pipeline.transforms[2].is_supervised = True

    @check_input_parameters_type()
    def before_train_iter(self, runner: BaseRunner):
        if self.interval > 1 and self.cnt == self.interval-1:
            # start supcon training
            # TODO : not using list index for stability
            dataset = self.get_dataset(runner)
            dataset.pipeline.transforms[2].is_supervised = False

        else:
            # TODO : not using list index for stability
            dataset = self.get_dataset(runner)
            dataset.pipeline.transforms[2].is_supervised = True

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        if self.interval > 1 and self.cnt < self.interval-1:
            self.cnt += 1

        elif self.interval > 1 and self.cnt == self.interval-1:
            # end supcon training
            # TODO : not using list index for stability
            dataset = self.get_dataset(runner)
            dataset.pipeline.transforms[2].is_supervised = True
            self.cnt = 0

