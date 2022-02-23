import numbers
import time
import torch
import os
import os.path as osp
import warnings

import torch.multiprocessing as mp
import torch.distributed as dist

from mmcv import __version__
from mmcv import collect_env
from mmcv import get_git_hash
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner
from mmcv.runner import load_checkpoint
from mmcv.runner import DistSamplerSeedHook

from mmcls import __version__ as mmcls_version
from mmdet import __version__ as mmdet_version
from mmdet.parallel import MMDataCPU

from mpa.registry import STAGES
from mpa.stage import Stage
from mpa.stage import _set_random_seed
from mpa.utils.convert_keys import convert_keys
from mpa.utils import logger
from mpa.utils.logger import get_logger

from .builder import build_backbone, build_trainer, build_dataset, build_dataloader


@STAGES.register_module()
class SelfSLTrainer(Stage):
    """
    Run a self supervised training with given backbone and dataset without annotation.
    The default selfsl method is 'BYOL' and base configuration file is provided by deafult.

    Args:
        cfg (str, optional): Path to config file of SelfSL.
        downstream_cfg (str, optional): Path to config file of downstream task.
        downstream_task (str, optional): Type of downstream task, e.g., classification.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.down_task = self.cfg.pop('down_task')
        self.distributed = False

    def _configure_data(self, data_cfg):
        """
        Configure dataset for SelfSL.
        The given dataset will be a dataource in dataset wrapper for SelfSL.

        Args:
            data_cfg (Config): An dataset config as datasource.
        """
        cfg = self.cfg
        cfg.data.train.down_task = self.down_task

        # get and put pipeline options from data_cfg to cfg. it will be used by Config.configure_data()
        # to update pipelines
        poptions = data_cfg['data'].get('pipeline_options')
        if poptions is not None:
            cfg.data['pipeline_options'] = poptions

        # TODO: enable to use data.train?
        datasource_cfg = data_cfg['data']['unlabeled']

        # Datasource config
        if not isinstance(datasource_cfg, (list, tuple)):
            datasource_cfg = [datasource_cfg]

        # remove pipelines in datasource
        for _cfg in datasource_cfg:
            _cfg['pipeline'] = []
            if self.down_task == 'detection':
                _cfg.filter_empty_gt = False

        cfg.data.train.datasource = datasource_cfg
        Stage.configure_data(cfg, True)

        return cfg

    def _configure_model(self, backbone_cfg, model_ckpt):
        """
        Configure model for SelfSL with the given backbone cfg.
        If the pretrained is NoneType, the ImageNet pretrained model provided by torch or mmcls
        will be loaded by default.

        Args:
            backbone_cfg (Config): A backbone config.
            model_ckpt (str): Path to pretrained model of backbone for warm starts.
                              Set 'random' to train from scratch.
        """
        cfg = self.cfg
        cfg.model.down_task = self.down_task

        # Load imagenet pretrained model by default for warm-start
        # resnet18/34/50/101/152, mobilenet_v2, ote_mobilenetv3_small/large/large_075, ote_efficientnet_b0
        # TODO: changed default init weight t2, ote_mobilenetv3_small/large/large_075, ote_efficientne
        backbone_type = backbone_cfg.type.lower()
        if 'ote' in backbone_type:
            cfg.model.pretrained = True
        elif 'resnet' in backbone_type:
            depth = backbone_cfg.depth
            if depth in [18, 34, 50, 101, 152]:
                cfg.model.pretrained = 'torchvision://resnet{}'.format(depth)
        elif 'mobilenet' in backbone_type:
            if 'v2' in backbone_type:
                cfg.model.pretrained = 'mmcls://mobilenet_v2'

        if not cfg.model.pretrained:
            logger.info(f'no pretrained model for {backbone_type}')

        if model_ckpt == 'random':
            logger.info('train from scratch')
            cfg.model.pretrained = None
        elif model_ckpt:
            if 'ote' in backbone_type:
                new_path = osp.join(cfg.work_dir, osp.basename(model_ckpt)[:-3] + 'converted.pth')\
                    if cfg.work_dir else None
                cfg.load_from = convert_keys(backbone_cfg.type, model_ckpt, new_path)
            else:
                cfg.load_from = model_ckpt
            cfg.model.pretrained = None

        # get the number of output channles of backbone
        backbone = build_backbone(backbone_cfg, self.down_task)
        output = backbone(torch.rand([1, 3, 224, 224]))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        out_channels = output.shape[1]

        # set in_channel of neck as out_channel of backbone
        if 'neck' in cfg.model:
            cfg.model.neck.in_channels = out_channels

        # unfreeze backbone and bn layers
        backbone_cfg['frozen_stages'] = -1
        if backbone_cfg.get('norm_eval'):
            backbone_cfg.norm_eval = False

        # change to syncBN in multi-gpu mode
        if self.distributed:
            # TODO: need an exeption handling for models w/o BN
            if 'resnet' in backbone_cfg.type:
                backbone_cfg.norm_cfg = dict(type='SyncBN')
            cfg.model.neck.norm_cfg = dict(type='SyncBN')
            cfg.model.head.predictor.norm_cfg = dict(type='SyncBN')

        cfg.model.backbone = backbone_cfg
        return cfg

    def configure(self, backbone_cfg, data_cfg, model_ckpt=None, resume=None):
        """
        Configure overall SelfSL pipeline with given backbone and dataset.

        Args:
            backbone_cfg (Config): A backbone config.
            data_cfg (Config): An dataset config.
            model_ckpt (str, optional): Path to pretrained model of backbone.
                                        Set 'random' to train from scratch.
            resume (str, optional): Path to model to resume from.
        """
        logger.info('called configure selfsl trainer')
        cfg = self.cfg

        # Distributed
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                self.distributed = True

        # Environment
        if not hasattr(cfg, 'deterministic'):
            cfg.deterministic = True
        _set_random_seed(cfg.seed, cfg.deterministic)
        logger.info(f'set random seed:{cfg.seed}, deterministic:{cfg.deterministic}')

        # Model
        self._configure_model(backbone_cfg, model_ckpt)

        if resume:
            cfg.resume_from = resume

        # Data
        self._configure_data(data_cfg)

        # add 'num_gpus' and 'batch_size' to the cfg
        if self.distributed:
            cfg.num_gpus = len(cfg.gpu_ids)
            cfg.batch_size = cfg.num_gpus * cfg.data.samples_per_gpu
        else:
            if torch.cuda.is_available():
                cfg.num_gpus = 1
            else:
                cfg.num_gpus = 0
            cfg.batch_size = cfg.data.samples_per_gpu

        # applying linear scaling rule
        cfg.base_lr = cfg.optimizer.lr
        cfg.optimizer.lr *= (cfg.batch_size / 256)
        logger.info(f'batch size is {cfg.batch_size} ({cfg.data.samples_per_gpu} x {max(cfg.num_gpus, 1)} gpus)')
        logger.info(f'scaling lr from {cfg.base_lr} to {cfg.optimizer.lr} following linear scaling rule')

        # TODO: only apply linear scaling rule if 'dist_params.linear_scale_lr' is set ???
        # if self.distributed:
        #    if cfg.dist_params.get('linear_scale_lr', False):
        #        new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
        #        logger.info(f'enabled linear scaling rule to the learning rate. \
        #            changed LR from {cfg.optimizer.lr} to {new_lr}')
        #        cfg.optimizer.lr = new_lr

        return cfg

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for selfsl

        - Configuration
        - Environment setup
        - Run training
        - Save backbone model as 'backbone.pth'
        """
        # Init logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self._init_logger()
        logger.info(f'called run with {kwargs}')
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        down_task = model_cfg.get('task', None)
        if down_task and down_task != self.down_task:
            logger.error(f'wrong {down_task} model for {self.down_task} task')
            raise ValueError()

        # Configure
        cfg = self.configure(model_cfg.model.backbone, data_cfg, model_ckpt)

        # Environment
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed
        meta['exp_name'] = cfg.model.type + cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmcv_version=__version__ + '.' + get_git_hash()[:7],
                mmdet_version=mmdet_version,
                mmcls_version=mmcls_version)

        # Save config
        cfg.dump(osp.join(cfg.work_dir, 'config.yaml'))
        logger.info(f'Config:\n{cfg.pretty_text}')

        # Data
        datasets = [build_dataset(cfg.data.train)]

        if self.distributed:
            os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = str(cfg.dist_params.get('master_port', '29500'))
            mp.spawn(SelfSLTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                     args=(datasets, cfg, self.distributed, timestamp, meta))
        else:
            SelfSLTrainer.train_worker(None, datasets, cfg,
                                       distributed=self.distributed,
                                       timestamp=timestamp,
                                       meta=meta)

        # Extract backbone weights
        chk = torch.load(osp.join(cfg.work_dir, 'latest.pth'))
        backbone_chk = dict(meta=chk['meta'], optimizer=chk['optimizer'], state_dict=dict())
        for k, v in chk['state_dict'].items():
            if k.startswith('online_backbone'):
                if 'OTE' in cfg.model.backbone.type:
                    backbone_chk['state_dict'][k[16:]] = v
                else:
                    backbone_chk['state_dict'][k[7:]] = v

        if len(backbone_chk['state_dict']) == 0:
            raise Exception('Cannot find a backbone module in the checkpoint')

        backbone_ckpt_path = osp.join(cfg.work_dir, 'backbone.pth')
        torch.save(backbone_chk, backbone_ckpt_path)

        return dict(pretrained=backbone_ckpt_path)

    @staticmethod
    def train_worker(gpu,
                     dataset,
                     cfg,
                     distributed=False,
                     timestamp=None,
                     meta=None):

        logger = get_logger()

        if distributed:
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend=cfg.dist_params.get('backend', 'nccl'),
                                    world_size=len(cfg.gpu_ids), rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        # Model
        model = build_trainer(cfg.model)

        # prepare data loaders
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                dist=distributed,
                seed=cfg.seed) for ds in dataset
        ]

        # put model on gpus
        if torch.cuda.is_available():
            if distributed:
                find_unused_parameters = cfg.get('find_unused_parameters', False)
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                model = MMDataParallel(
                    model.cuda(), device_ids=[torch.cuda.current_device()])
        else:
            model = MMDataCPU(model)

        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)

        if cfg.get('runner') is None:
            cfg.runner = {
                'type': 'EpochBasedRunner',
                'max_epochs': cfg.total_epochs
            }
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

        # an ugly walkaround to make the .log and .log.json filenames the same
        runner.timestamp = timestamp

        # register hooks
        runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                       cfg.checkpoint_config, cfg.log_config,
                                       cfg.get('momentum_config', None))

        # When distributed training, it is only useful in conjunction with 'EpochBasedRunner`,
        # while `IterBasedRunner` achieves the same purpose with `IterLoader`.
        if distributed and cfg.runner.type == 'EpochBasedRunner':
            runner.register_hook(DistSamplerSeedHook())

        for hook in cfg.get('custom_hooks', ()):
            runner.register_hook_from_cfg(hook)

        if cfg.get('resume_from', False):
            runner.resume(cfg.resume_from)
        elif cfg.get('load_from', False):
            logger.info(f'load checkpoint from {cfg.load_from}')
            try:
                # workaround code to reduce a bunch of logs about missing keys during loading checkpoint
                _backbone = torch.nn.Module()
                _backbone.add_module(
                    'backbone',
                    getattr(runner.model, 'module', runner.model).online_backbone)
                load_checkpoint(_backbone, cfg.load_from, map_location='cpu', strict=False, logger=logger)
            except IOError:
                logger.warn('cannot open {}'.format(cfg.load_from))

        runner.run(data_loaders, cfg.workflow)
