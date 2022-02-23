import os
import copy
# import random
import pickle
import numpy as np
# import torch
# from torch.utils.data import Dataset

import torchvision

from mmcv.utils.registry import build_from_cfg
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.pipelines import Compose

# from .sources.builder import build_datasource
from .sources.index_list import ImageIndexList

from mpa.utils import logger


@DATASETS.register_module()
class TVDatasetSplit(BaseDataset):
    """ create a dataset as a subset from torchvision dataset using
        include and/or exclude image indices (list or pickle file).
    """
    def __init__(self, base, data_prefix=None, **kwargs):
        num_images = kwargs.pop('num_images', -1)
        pipeline = kwargs.pop('pipeline', None)
        num_classes = kwargs.pop('num_classes', 10)
        samples_per_gpu = kwargs.pop('samples_per_gpu', 1)
        workers_per_gpu = kwargs.pop('workers_per_gpu', 1)
        include_idx = kwargs.pop('include_idx', None)
        exclude_idx = kwargs.pop('exclude_idx', None)
        classes = kwargs.pop('classes', range(num_classes))
        use_labels = kwargs.pop('use_labels', True)
        _ = kwargs.pop('test_mode', False)
        seed = kwargs.pop("seed", None)
        if seed is not None:
            np.random.seed(seed)

        if 'download' not in kwargs:
            kwargs['download'] = True

        if isinstance(base, str):
            if not data_prefix:
                data_prefix = os.path.join('data/torchvision', base.lower())
            self.base = getattr(torchvision.datasets, base)(data_prefix, **kwargs)
        else:
            self.base = base

        if hasattr(self.base, 'targets'):
            self.labels = np.array(self.base.targets)
        elif hasattr(self.base, 'labels'):
            self.labels = np.array(self.base.labels)
        else:
            raise NotImplementedError(f'check the attribute name of torchvision \
                                      dataset {base}')

        indices_pool = range(len(self.labels))
        if include_idx is not None:
            if isinstance(include_idx, list):
                indices_pool = include_idx
            elif isinstance(include_idx, str):
                if os.path.exists(include_idx):
                    indices_pool = pickle.load(open(include_idx, 'rb'))
                else:
                    logger.warning(f'cannot find include index pickle file \
                                    {include_idx}. ignored.')
            else:
                raise TypeError(f"not supported type for 'include_idx'.\
                        should be list or pickle file path but {type(include_idx)}")

        if exclude_idx is not None:
            if isinstance(exclude_idx, str):
                if os.path.exists(exclude_idx):
                    exclude_idx = pickle.load(open(exclude_idx, 'rb'))
                else:
                    logger.warning(f'cannot find exclude index pickle file \
                                {exclude_idx}. ignored.')
            elif not isinstance(exclude_idx, list):
                raise TypeError(f"not supported type for 'exclude_idx'.\
                    should be list or pickle file path but {type(exclude_idx)}")
        if isinstance(exclude_idx, list):
            indices_pool = np.setdiff1d(indices_pool, exclude_idx)
        self.exclude_idx = []
        if exclude_idx is not None:
            self.exclude_idx = exclude_idx

        self.CLASSES = classes
        self.num_classes = num_classes
        self._samples_per_gpu = samples_per_gpu
        self._workers_per_gpu = workers_per_gpu
        indices = []
        if num_images != -1:
            if num_images > len(indices_pool) or num_images < 0:
                raise RuntimeError(f"cannot generate split dataset. \
                    length of base dataset = {len(indices_pool)}, \
                    requested split {num_images}")
            if use_labels:
                items_per_class = num_images // self.num_classes
                for i in classes:
                    idx = np.where(self.labels == i)[0]
                    idx = list(set(idx) & set(indices_pool))
                    idx = np.random.choice(idx, items_per_class, False)
                    np.random.shuffle(idx)
                    indices.extend(idx)
                indices = np.array(indices)
            else:
                indices = np.random.choice(indices_pool, num_images, False)
        else:
            for i in classes:
                idx = np.where(self.labels == i)[0]
                idx = list(set(idx) & set(indices_pool))
                np.random.shuffle(idx)
                indices.extend(idx)
            indices = np.array(indices)

        self.indices = indices
        self.data_source = ImageIndexList(self.base, indices)

        self.num_pipes = 1
        if pipeline is None:
            # set default pipeline
            pipeline = [
                dict(type='Normalize', mean=[127., 127., 127.], std=[127., 127., 127.]),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='ToTensor', keys=['gt_label']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]

        if isinstance(pipeline, dict):
            self.pipeline = {}
            for k, v in pipeline.items():
                self.pipeline[k] = \
                    Compose([build_from_cfg(p, PIPELINES) for p in v])
                logger.debug(self.pipeline[k])
            self.num_pipes = len(pipeline)
        elif isinstance(pipeline, list):
            if len(pipeline) <= 0:
                self.pipeline = None
            else:
                self.pipeline = \
                        Compose([build_from_cfg(p, PIPELINES) for p in pipeline])

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        logger.info(f'called {__name__}')
        data_infos = []
        for idx in range(self.data_source.get_length()):
            img, label = self.data_source.get_sample(idx)
            info = {'img': img, 'gt_label': np.array(label, dtype=np.int64)}
            data_infos.append(info)
        logger.info(f'prepared {len(data_infos)} datapoints')
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.pipeline is None:
            return self.data_infos[idx]

        data_infos = [
            copy.deepcopy(self.data_infos[idx]) for _ in range(self.num_pipes)
        ]
        if isinstance(self.pipeline, dict):
            results = {}
            for i, (k, v) in enumerate(self.pipeline.items()):
                results[k] = self.pipeline[k](data_infos[i])
        else:
            results = self.pipeline(data_infos[0])

        return results

    @property
    def samples_per_gpu(self):
        return self._samples_per_gpu

    @property
    def workers_per_gpu(self):
        return self._workers_per_gpu
