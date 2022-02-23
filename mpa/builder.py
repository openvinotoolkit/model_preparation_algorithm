import os
import time
import copy

from mmcv import Config
from mmcv import ConfigDict
from mmcv import build_from_cfg
# from collections import defaultdict

from .workflow import Workflow
from mpa.modules.hooks.workflow_hook import build_workflow_hook

from mpa.utils.logger import config_logger
from mpa.utils import logger

from mpa.registry import STAGES
from mpa.stage import get_available_types


def __build_stage(config, common_cfg=None, index=0):
    logger.info(f'called build_stage({config, common_cfg})')
    config.type = config.type if 'type' in config.keys() else 'Stage'  # TODO: tmp workaround code for competability
    config.common_cfg = common_cfg
    config.index = index
    return build_from_cfg(config, STAGES)


def __build_workflow(config):
    logger.info(f'called build_workflow({config})')

    whooks = []
    whooks_cfg = config.get('workflow_hooks', [])
    for whook_cfg in whooks_cfg:
        whook = build_workflow_hook(whook_cfg.copy())
        whooks.append(whook)

    output_path = config.get('output_path', 'logs')
    folder_name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    config.output_path = os.path.join(output_path, folder_name)
    os.makedirs(config.output_path, exist_ok=True)

    # create symbolic link to the output path
    symlink_dst = os.path.join(output_path, 'latest')
    if os.path.exists(symlink_dst):
        os.unlink(symlink_dst)
    os.symlink(folder_name, symlink_dst, True)

    log_level = config.get('log_level', 'INFO')
    config_logger(os.path.join(config.output_path, 'app.log'), level=log_level)

    common_cfg = copy.deepcopy(config)
    common_cfg.pop('stages')
    if len(whooks_cfg) > 0:
        common_cfg.pop('workflow_hooks')

    stages = [__build_stage(stage_cfg.copy(), common_cfg, index=i) for i, stage_cfg in enumerate(config.stages)]
    return Workflow(stages, whooks)


def build(config, mode=None, stage_type=None):
    logger.info(f'called build_recipe({config})')

    if not isinstance(config, Config):
        if isinstance(config, str):
            if os.path.exists(config):
                config = Config.fromfile(config)
            else:
                logger.error(f'cannot find configuration file {config}')
                raise ValueError(f'cannot find configuration file {config}')

    if hasattr(config, 'stages'):
        # build as workflow
        return __build_workflow(config)
    else:
        # build as stage
        if not hasattr(config, 'type'):
            logger.info('seems to be passed stage yaml...')
            supported_stage_types = get_available_types()
            if stage_type in supported_stage_types:
                cfg_dict = ConfigDict(
                    dict(
                        type=stage_type,
                        name='default',
                        mode=mode,
                        config=config,
                        index=0
                    )
                )
            else:
                msg = f'type {stage_type} is not in {supported_stage_types}'
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            pass
        return __build_stage(cfg_dict)
