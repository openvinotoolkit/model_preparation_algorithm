from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class StopHook(Hook):
    def __init__(self, stop_point=15000, **kwargs):
        super(StopHook, self).__init__(**kwargs)
        self.stop_point = stop_point

    def before_train_iter(self, runner):
        if runner.iter == self.stop_point:
            runner._iter = runner._max_iters
