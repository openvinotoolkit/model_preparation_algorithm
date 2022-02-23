from mmcv.runner.hooks import HOOKS
from mmdet.apis.ote.extension.utils.hooks import EarlyStoppingHook


@HOOKS.register_module()
class LazyEarlyStoppingHook(EarlyStoppingHook):
    def __init__(self,
                 interval: int,
                 metric: str = 'bbox_mAP',
                 rule: str = None,
                 patience: int = 5,
                 iteration_patience: int = 500,
                 min_delta: float = 0.0,
                 start: int = None):
        self.start = start
        super(LazyEarlyStoppingHook, self).__init__(interval, metric, rule, patience, iteration_patience, min_delta)

    def _should_check_stopping(self, runner):
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            return False
        else:
            if (current + 1 - self.start) % self.interval:
                return False
        return True
