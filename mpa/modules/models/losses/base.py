from abc import ABCMeta, abstractmethod

import torch.nn as nn

# from mmseg.core import build_pixel_sampler
from ...utils.seg_sampler import build_pixel_sampler

from .utils_pixel_wise import builder


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float or dict): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, ignore_index=255, sampler=None, **kwargs):
        super().__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

        self.sampler = None
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, ignore_index=ignore_index)

        self._loss_weight_scheduler = builder.build_scheduler(loss_weight, default_value=1.0)

        self._iter = 0
        self._last_loss_weight = 0
        self._epoch_size = 1

    def set_step_params(self, init_iter, epoch_size):
        assert init_iter >= 0
        assert epoch_size > 0

        self._iter = init_iter
        self._epoch_size = epoch_size

    @property
    def iter(self):
        return self._iter

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def last_loss_weight(self):
        return self._last_loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.

        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.

        Returns:
            torch.Tensor: The calculated loss.
        """

        self._last_loss_weight = self._loss_weight_scheduler(self.iter, self.epoch_size)

        loss, meta = self._forward(*args, **kwargs)
        out_loss = self._last_loss_weight * loss

        self._iter += 1

        return out_loss, meta
