import torch
from mmcv.runner import HOOKS, Hook
from mmcv.parallel import is_module_wrapper
from mpa.utils import logger


@HOOKS.register_module()
class DualModelEMAHook(Hook):
    """Generalized re-implementation of mmcv.runner.EMAHook

    Source model paramters would be exponentially averaged
    onto destination model pararmeters on given intervals

        .. math::

            \text{Xema_{t+1}} = (1 - \text{momentum}) \times
            \text{Xema_{t}} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        start_epoch (int): During initial a few epochs, we just copy values
            to update ema parameters. Defaults to 5.
        src_model_name (str): Source model for EMA (X)
        dst_model_name (str): Destination model for EMA (Xema)
    """

    def __init__(
        self,
        momentum=0.0002,
        interval=1,
        start_epoch=5,
        src_model_name='model_s',
        dst_model_name='model_t',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.momentum = momentum**interval
        self.interval = interval
        self.start_epoch = start_epoch
        self.src_model_name = src_model_name
        self.dst_model_name = dst_model_name
        self.enabled = False

    def before_run(self, runner):
        """Set up src & dst model parameters."""
        model = self._get_model(runner)
        self.src_model = getattr(model, self.src_model_name, None)
        self.dst_model = getattr(model, self.dst_model_name, None)
        if self.src_model and self.dst_model:
            self.enabled = True
            self.src_params = self.src_model.state_dict(keep_vars=True)
            self.dst_params = self.dst_model.state_dict(keep_vars=True)
            if runner.epoch == 0 and runner.iter == 0:
                # If it's not resuming from a checkpoint
                # initialize student model by teacher model
                # (teacher model is main target of load/save)
                # (if it's resuming there will be student weights in checkpoint. No need to copy)
                self._sync_model()
                logger.info('Initialized student model by teacher model')

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if not self.enabled:
            return

        if runner.iter % self.interval != 0:
            # Skip update
            return

        if runner.epoch + 1 < self.start_epoch:
            # Just copy parameters before start epoch
            self._copy_model()
            return

        # EMA
        self._ema_model()

    def after_train_epoch(self, runner):
        if self.enabled:
            logger.info(f'src dst diff: {self._diff_model()}')

    def _get_model(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        return model

    def _sync_model(self):
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                dst_param = self.dst_params[name]
                src_param.data.copy_(dst_param.data)

    def _copy_model(self):
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                dst_param = self.dst_params[name]
                dst_param.data.copy_(src_param.data)

    def _ema_model(self):
        momentum = min(self.momentum, 1.0)
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                dst_param = self.dst_params[name]
                # dst_param.data.mul_(1 - momentum).add_(src_param.data, alpha=momentum)
                dst_param.data.copy_(dst_param.data*(1 - momentum) + src_param.data*momentum)

    def _diff_model(self):
        diff_sum = 0.0
        with torch.no_grad():
            for name, src_param in self.src_params.items():
                dst_param = self.dst_params[name]
                diff = ((src_param - dst_param)**2).sum()
                diff_sum += diff
        return diff_sum
