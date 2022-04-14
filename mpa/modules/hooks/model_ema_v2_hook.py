# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import HOOKS, Hook
from mmcv.parallel import is_module_wrapper

from copy import deepcopy
import torch.nn as nn

from mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class ModelEmaV2Hook(Hook):
    """
    Source model paramters would be exponentially averaged
    onto destination model pararmeters on given intervals

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        ema_decay (float): EMA decay value used for updating ema parameter.
            Defaults to 0.999.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        start_epoch (int): During initial a few epochs, we just copy values
            to update ema parameters. Defaults to 5.
    """

    def __init__(self, ema_decay=0.999, interval=1, start_epoch=5, **kwargs):
        super().__init__(**kwargs)
        self.ema_decay = ema_decay
        self.should_save_ema_model = False
        self.interval = interval
        self.start_epoch = start_epoch

    def before_run(self, runner):
        """Set up src & dst model parameters."""
        model = self._get_model(runner)
        ema_model = ModelEmaV2(model, decay=self.ema_decay)
        runner.ema_model = ema_model
        runner.use_ema = True
        logger.info("\t* EMA V2 Enable")

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if runner.iter % self.interval != 0:
            # Skip update
            return

        if runner.epoch < self.start_epoch:
            # Just copy parameters before start epoch
            return
        # EMA
        runner.ema_model.update()

    def _get_model(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model


class ModelEmaV2(nn.Module):
    """Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.src_model = model.state_dict()
        self.dst_model = self.module.state_dict()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.dst_model.values(), self.src_model.values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self):
        self._update(
            update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )
