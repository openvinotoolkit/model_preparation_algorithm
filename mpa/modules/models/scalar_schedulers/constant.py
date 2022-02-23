# from ..builder import SCALAR_SCHEDULERS
from ..losses.utils_pixel_wise.builder import SCALAR_SCHEDULERS
from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class ConstantScalarScheduler(BaseScalarScheduler):
    def __init__(self, scale=30.0):
        super(ConstantScalarScheduler, self).__init__()

        self._end_s = scale
        assert self._end_s > 0.0

    def _get_value(self, step, epoch_size):
        return self._end_s
