import warnings

# from mmcv.cnn import MODELS as MMCV_MODELS
# from mmcv.utils import Registry
# by han
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import Sequential
from .registry import Registry, build_from_cfg


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


MMCV_MODELS = Registry('model', build_func=build_model_from_cfg)
MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
PARAMS_MANAGERS = MODELS
SCALAR_SCHEDULERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg, ignore_index=255):
    """Build loss."""
    return LOSSES.build(cfg, default_args=dict(ignore_index=ignore_index))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    return SEGMENTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_params_manager(cfg):
    return PARAMS_MANAGERS.build(cfg)


def build_scheduler(cfg, default_value=None):
    if cfg is None:
        if default_value is not None:
            assert isinstance(default_value, (int, float))
            cfg = dict(type='ConstantScalarScheduler', scale=float(default_value))
        else:
            return None
    elif isinstance(cfg, (int, float)):
        cfg = dict(type='ConstantScalarScheduler', scale=float(cfg))

    return SCALAR_SCHEDULERS.build(cfg)
