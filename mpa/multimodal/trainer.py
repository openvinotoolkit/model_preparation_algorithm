import os
import os.path as osp
import time
import numbers
import warnings

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner, HOOKS
from mmcv.runner import load_checkpoint
from mmcv.runner import DistSamplerSeedHook

from mmcls import __version__
from mmcls.utils import collect_env
from mmcls.core import (DistOptimizerHook, Fp16OptimizerHook)

from mmdet.parallel import MMDataCPU

from mpa.registry import STAGES
from mpa.multimodal.stage import MultimodalStage
from mpa.utils.logger import get_logger
from mpa.modules.hooks.multimodal_eval_hook import MultimodalEvalHook

from .builder import build_model, build_dataset, build_dataloader

SUPPORTED_MODALITY_TYPES = ['vision', 'tabular']
logger = get_logger()

@STAGES.register_module()
class MultimodalTrainer(MultimodalStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}
        
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        
        # Environment
        distributed = False
        if cfg.gpu_ids is not None:
            if isinstance(cfg.get('gpu_ids'), numbers.Number):
                cfg.gpu_ids = [cfg.get('gpu_ids')]
            if len(cfg.gpu_ids) > 1:
                distributed = True

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)

        
        cfg.data.train.modalities = check_modalities(cfg.data.train, phase='train')
        datasets = [build_dataset(cfg.data.train)]
        cfg.model.tabular_encoder.in_channels = datasets[0].table_df.shape[1]
        
        # Dataset for HPO
        hp_config = kwargs.get('hp_config', None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)
        
        
        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed

        if isinstance(datasets[0], list):
            repr_ds = datasets[0][0]
        else:
            repr_ds = datasets[0]
        
        cfg.model.head.n_classes = len(repr_ds.CLASSES)
                                                      
        if distributed:
            os.environ['MASTER_ADDR'] = cfg.dist_params.get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = cfg.dist_params.get('master_port', '29500')

            mp.spawn(MultimodalTrainer.train_worker, nprocs=len(cfg.gpu_ids),
                     args=(datasets, cfg, distributed, True, timestamp, meta))
        else:
            MultimodalTrainer.train_worker(None, datasets, cfg,
                                    distributed,
                                    True,
                                    timestamp,
                                    meta)
        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, 'best_model.pth'
                                    if osp.exists(osp.join(cfg.work_dir, 'best_model.pth'))
                                    else 'latest.pth')
        
        return dict(final_ckpt=output_ckpt_path)

    @staticmethod
    def train_worker(gpu, dataset, cfg, distributed, validate, timestamp, meta):
        if distributed:
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend=cfg.dist_params.get('backend', 'nccl'),
                                    world_size=len(cfg.gpu_ids), rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        # model
        cfg.model.modalities = cfg.data.train.modalities
        model = build_model(cfg.model)
        
        # prepare data loaders
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                dist=distributed,
                sample=False,
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
        runner.timestamp = f'{timestamp}' 

        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config
        
        # register hooks
        runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                       cfg.checkpoint_config, cfg.log_config, 
                                       cfg.get('momentum_config', None))
            
        # When distributed training, it is only useful in conjunction with 'EpochBasedRunner`,
        # while `IterBasedRunner` achieves the same purpose with `IterLoader`.
        if distributed and cfg.runner.type == 'EpochBasedRunner':
            runner.register_hook(DistSamplerSeedHook())

        for hook in cfg.get('custom_hooks', ()):
            runner.register_hook_from_cfg(hook)
        
        # register eval hooks
        if validate:
            cfg.data.val.modalities = check_modalities(cfg.data.val, phase='val')
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False
            )
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = MultimodalEvalHook #TODO: support for distributed environment
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='HIGHEST')
        
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
    
def check_modalities(cfg, phase):
    modalities = cfg.get('modalities', {})
    for modality in modalities:
        if modality not in SUPPORTED_MODALITY_TYPES:
            raise NotImplementedError(
                f"{modality} is not supported modality type. Currently \
                supported types are [{SUPPORTED_MODALITY_TYPES}]")

    logger.info('Used modal for {}: {} '.format(phase, modalities))
    assert len(modalities) > 0, "At least 1 modality is needed for training"
    
    return modalities