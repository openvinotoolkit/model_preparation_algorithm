import os
import copy
import pickle
import numpy as np

import torchvision

from mmcv.utils.registry import build_from_cfg
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.pipelines import Compose

from mpa.modules.datasets.sources.index_list import ImageIndexList

from mpa.utils import logger


@DATASETS.register_module()
class ClsTVDataset(BaseDataset):
    """ create a dataset as a subset from torchvision dataset using
        include and/or exclude image indices (list or pickle file).
    """
    def __init__(self, base, classes=[], new_classes=[], data_prefix=None, **kwargs):
        self.num_images = kwargs.pop('num_images', -1)
        self._samples_per_gpu = kwargs.pop('samples_per_gpu', 1)
        self._workers_per_gpu = kwargs.pop('workers_per_gpu', 1)
        pipeline = kwargs.pop('pipeline', None)
        include_idx = kwargs.pop('include_idx', None)
        exclude_idx = kwargs.pop('exclude_idx', None)
        self.use_labels = kwargs.pop('use_labels', True)
        self.balanced_class = kwargs.pop('balanced_class', True)
        _ = kwargs.pop('test_mode', False)
        self.dataset_type = base
        self.class_acc = False
        self.img_indices = dict(old=[], new=[])

        if 'download' not in kwargs:
            kwargs['download'] = True

        # Get Dataset from torchvision.datasets
        if isinstance(base, str):
            if not data_prefix:
                data_prefix = os.path.join('data/torchvision', base.lower())
            self.base = getattr(torchvision.datasets, base)(data_prefix, **kwargs)
        else:
            self.base = base

        # Get Labels from torchvision.datasets
        if hasattr(self.base, 'targets'):
            self.labels = np.array(self.base.targets)
        elif hasattr(self.base, 'labels'):
            self.labels = np.array(self.base.labels)
        else:
            raise NotImplementedError(f'check the attribute name of torchvision \
                                      dataset {base}')
        if not classes:
            classes = list(set(self.labels.tolist()))
        self.CLASSES = classes
        self.new_classes = new_classes

        # Configuration indices pool
        indices_pool = range(len(self.labels))
        indices_pool, self.exclude_idx = self.configure_idx(indices_pool, include_idx, exclude_idx)

        # Sampling indices
        self.indices = self.sampling_idx(indices_pool)
        self.data_source = ImageIndexList(self.base, self.indices)

        # configuration Pipelines
        self.pipeline, self.num_pipes = self.configure_pipeline(pipeline)

        # Load Annotations
        self.data_infos = self.load_annotations()
        self.statistics()

    def statistics(self):
        logger.info(f'ClsTVDataset - {len(self.CLASSES)} classes from {self.dataset_type}')
        logger.info(f'- Classes: {self.CLASSES}')
        if self.new_classes:
            logger.info(f'- New Classes: {self.new_classes}')
            old_data_length = len(self.img_indices['old'])
            new_data_length = len(self.img_indices['new'])
            logger.info(f'- # of old classes images: {old_data_length}')
            logger.info(f'- # of New classes images: {new_data_length}')
        logger.info(f'- # of images: {len(self)}')

    @staticmethod
    def configure_idx(indices_pool, include_idx, exclude_idx):
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
        return indices_pool, exclude_idx

    def sampling_idx(self, indices_pool):
        indices = []
        if self.num_images != -1:
            if self.num_images > len(indices_pool) or self.num_images < 0:
                raise RuntimeError(f"cannot generate split dataset. \
                    length of base dataset = {len(indices_pool)}, \
                    requested split {self.num_images}")
            # if self.balanced_class:
            items_per_class = self.num_images // len(self.CLASSES)
            for i in self.CLASSES:
                idx = np.where(self.labels == i)[0]
                idx = list(set(idx) & set(indices_pool))
                if self.balanced_class:
                    idx = np.random.choice(idx, items_per_class, False)
                    np.random.shuffle(idx)
                indices.extend(idx)
            indices = np.array(indices)
            if not self.balanced_class:
                indices = np.random.choice(indices, self.num_images, False)
        else:
            for i in self.CLASSES:
                idx = np.where(self.labels == i)[0]
                idx = list(set(idx) & set(indices_pool))
                np.random.shuffle(idx)
                indices.extend(idx)
            indices = np.array(indices)
        return indices

    @staticmethod
    def configure_pipeline(pipeline_cfg):
        num_pipes = 1
        pipeline = {}
        if pipeline_cfg is None:
            # set default pipeline
            pipeline_cfg = [
                dict(type='Normalize', mean=[127., 127., 127.], std=[127., 127., 127.]),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='ToTensor', keys=['gt_label']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]

        if isinstance(pipeline_cfg, dict):
            for k, v in pipeline_cfg.items():
                pipeline[k] = \
                    Compose([build_from_cfg(p, PIPELINES) for p in v])
                logger.debug(pipeline[k])
            num_pipes = len(pipeline_cfg)
        elif isinstance(pipeline_cfg, list):
            if len(pipeline_cfg) <= 0:
                pipeline = None
            else:
                pipeline = \
                        Compose([build_from_cfg(p, PIPELINES) for p in pipeline_cfg])
        return pipeline, num_pipes

    def load_annotations(self):
        data_infos = []
        for idx in range(self.data_source.get_length()):
            img, label = self.data_source.get_sample(idx)
            gt_label = self.class_to_idx[label]
            info = {'img': img, 'gt_label': np.array(gt_label, dtype=np.int64)}
            data_infos.append(info)
            if label in self.new_classes:
                self.img_indices['new'].append(idx)
            else:
                self.img_indices['old'].append(idx)
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

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset with new metric 'class_accuracy'

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
                'accuracy', 'precision', 'recall', 'f1_score', 'support', 'class_accuracy'
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5) if len(self.CLASSES) >= 5 else (1, )}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric

        if 'class_accuracy' in metrics:
            metrics.remove('class_accuracy')
            self.class_acc = True

        eval_results = super().evaluate(results, metrics, metric_options, logger)

        # Add Evaluation Accuracy score per Class
        if self.class_acc:
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            accuracies = self.class_accuracy(results, gt_labels)
            eval_results.update({f'{c} accuracy': a for c, a in zip(self.CLASSES, accuracies)})
            eval_results.update({'mean accuracy': np.mean(accuracies)})

        return eval_results

    def class_accuracy(self, results, gt_labels):
        accracies = []
        pred_label = results.argsort(axis=1)[:, -1:][:, ::-1]
        for i in range(len(self.CLASSES)):
            cls_pred = pred_label == i
            cls_pred = cls_pred[gt_labels == i]
            cls_acc = np.sum(cls_pred) / len(cls_pred)
            accracies.append(cls_acc)
        return accracies

    @property
    def samples_per_gpu(self):
        return self._samples_per_gpu

    @property
    def workers_per_gpu(self):
        return self._workers_per_gpu
