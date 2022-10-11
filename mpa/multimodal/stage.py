import copy
import torch
import numpy as np

from mmcv import ConfigDict
from mmcv import build_from_cfg

from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger

logger = get_logger()

class MultimodalStage(Stage):
    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure: training={training}')

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                cfg.model = copy.deepcopy(model_cfg.model)

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)

        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        self.configure_data(cfg, training, **kwargs)

        # Other hyper-parameters
        if cfg.get('hyperparams', False):
            self.configure_hyperparams(cfg, training, **kwargs)

        return cfg


    @staticmethod
    def configure_hyperparams(cfg, training, **kwargs):
        hyperparams = kwargs.get('hyperparams', None)
        if hyperparams is not None:
            bs = hyperparams.get('bs', None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get('lr', None)
            if lr is not None:
                cfg.optimizer.lr = lr