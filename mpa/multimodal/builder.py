import random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, WeightedRandomSampler

from mmcv import build_from_cfg
from mmcv.utils import Registry
from mmcv.parallel import collate

from functools import partial
from mmcv.runner import get_dist_info

from mpa.modules.datasets.samplers.distributed_sampler import DistributedSampler

STAGES = Registry('stages')
MODELS = Registry('models')
BACKBONES = Registry('backbones')
HEADS = Registry('heads')
NECKS = Registry('necks')
DATASETS = Registry('datasets')

CLASS_0 = 26030
CLASS_1 = 470
SUBSET = 1/5

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_model(cfg):
    return build(cfg, MODELS)

def build_pipelines(cfg):
    from mmcls.datasets.builder import PIPELINES
    return build(cfg, PIPELINES)

def build_dataset(cfg, default_args=None):
    return build(cfg, DATASETS, default_args)

def build_loss(cfg):
    """Build loss."""
    from mmcls.models.builder import LOSSES
    return build(cfg, LOSSES)

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     sample=False,
                     shuffle=True,
                     seed=None,
                     args=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        sample (bool): Use of sampler for balancing thhe dataset. Default: False.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    elif sample:
        class_weights = [1 , CLASS_0/CLASS_1 ]  # TODO: make them adjustable
        img_weights = [0]*len(dataset)
        for idx, data in enumerate(dataset):
            label = data['gt_label']
            img_weights[idx] = class_weights[label]
        num_samples = int(len(img_weights)*SUBSET)
        sampler = WeightedRandomSampler(img_weights, num_samples=num_samples, replacement=True)
        shuffle = False
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_backbone(cfg):
    modality = cfg.pop('modal')
    if modality == 'vision':
        from mmcls.models.builder import BACKBONES
    elif modality == 'text':
        #TODO: implement
        pass
    elif modality == 'tabular':
        from mpa.multimodal.builder import BACKBONES
    else:
        ValueError('{} is not supported modality'.format(modality))
    return build(cfg, BACKBONES)

def build_neck(cfg):
    return build(cfg, NECKS)

def build_head(cfg):
    return build(cfg, HEADS)
