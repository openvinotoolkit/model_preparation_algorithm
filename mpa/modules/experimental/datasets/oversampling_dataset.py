import os
import cv2
import tqdm
import numpy as np
import pandas as pd

import torch
from mmseg.datasets import DATASETS, build_dataset
from mpa.utils import logger


@DATASETS.register_module()
class OversamplingSampler(object):
    """Dataset wrapper for oversampling for Imbalanced dataset.
    """

    def __init__(self,
                 orig_type=None,
                 sampling_mode='whole',
                 weights_mode='max',
                 weights_file='oversampling_weights.csv',
                 **kwargs):

        assert orig_type is not None
        assert sampling_mode in ['whole', 'half', 'strict_half']
        assert weights_mode in ['max', 'sum', 'adaptive']

        if sampling_mode == 'strict_half':
            logger.info("In 'strict_half' sampling mode, OverSamplingSegmentor and half batch_size are required.")
            logger.info("If samples_per_gpu is set to 8, REAL samples_per_gpu will be 16.")

        self.sampling_mode = sampling_mode

        dataset_cfg = kwargs.copy()
        dataset_cfg['type'] = orig_type
        self.dataset = build_dataset(dataset_cfg)
        self.extension = self.dataset.img_infos[0]['filename'].split('.')[-1]

        if os.path.isfile(os.path.join(kwargs['data_root'], weights_file)):
            self.df_dataset = pd.read_csv(os.path.join(kwargs['data_root'], weights_file),
                                          index_col='Unnamed: 0')

            if not [self.dataset.img_infos[i]['filename']
                    for i in range(len(self.dataset.img_infos))] == self.df_dataset.index.tolist():

                logger.info(("The order of self.dataset and the previously saved order of "
                             "self.df_dataset do not match."))
                logger.info("self.df_dataset is reset.")
                self.set_df()
                self.df_dataset.to_csv(os.path.join(kwargs['data_root'], weights_file))
                logger.info("self.df_dataset is saved at {}.".format(
                    os.path.join(kwargs['data_root'], weights_file)))

        else:
            logger.info("self.df_dataset is set.")
            self.set_df()
            self.df_dataset.to_csv(os.path.join(kwargs['data_root'], weights_file))
            logger.info("self.df_dataset is saved at {}.".format(
                os.path.join(kwargs['data_root'], weights_file)))

        self.num_samples = len(self.dataset)

        self.init_weights(weights_mode=weights_mode)
        self.init_counts()

    def set_df(self):
        # ann_list = os.listdir(self.dataset.ann_dir)
        # ann_extension = ann_list[0].split('.')[-1]
        ann_infos = {}
        for path in tqdm.tqdm(self.dataset.img_infos, desc='Get # of classes', total=len(self.dataset.img_infos)):
            img_path, ann_path = path['filename'], path['ann']['seg_map']
            if img_path not in ann_infos:
                # img_path = path.replace(ann_extension, self.extension)
                ann_infos[img_path] = \
                    [0 for _ in range(len(self.dataset.CLASSES))]
                mask = cv2.imread(os.path.join(self.dataset.ann_dir, ann_path))
                labels = np.unique(mask)
                for label in labels:
                    if label != 255:
                        ann_infos[img_path][label] = 1

        # ann_infos = dict(sorted(ann_infos.items(), key=lambda x: x[0]))
        self.df_dataset = pd.DataFrame(ann_infos).T

    def init_weights(self, weights_mode='max'):
        weights = (1 / self.df_dataset.sum(axis=0)).tolist()
        if weights_mode == 'max':
            self.df_dataset['weights'] = 0
            for i in range(len(self.dataset.CLASSES)):
                self.df_dataset['weights'] = [
                    max(*elem) for elem in zip(
                        self.df_dataset['weights'],
                        self.df_dataset.iloc[:, i].apply(lambda x: weights[i] if x else 0)
                    )
                ]

        elif weights_mode == 'sum':
            raise NotImplementedError()
            # for i in range(self.num_samples):
            #     if i == 0:
            #         result = self.df_dataset.loc[:,i].apply(
            #             lambda x: weights[i] if x else 0)
            #     else:
            #         result += self.df_dataset.loc[:,i].apply(
            #             lambda x: weights[i] if x else 0)
            # self.df_dataset['weights'] = result

        elif weights_mode == 'adaptive':
            raise NotImplementedError()

        self.weights = torch.DoubleTensor(self.df_dataset['weights'].tolist())

    def init_counts(self):
        self.cnt_samples = 0
        self.idx_sampling = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()

    def __len__(self):
        """Total number of samples of data."""
        return self.num_samples

    def __getitem__(self, idx):
        if self.sampling_mode == 'whole':
            new_idx = self.idx_sampling[idx]
            self.cnt_samples += 1
            if self.cnt_samples >= self.num_samples:
                self.init_counts()

            return self.dataset.__getitem__(new_idx)

        elif self.sampling_mode == 'half':
            if torch.rand(1) > .5:
                return self.dataset.__getitem__(idx)
            else:
                new_idx = self.idx_sampling[idx]
                self.cnt_samples += 1
                if self.cnt_samples >= self.num_samples:
                    self.init_counts()

                return self.dataset.__getitem__(new_idx)

        elif self.sampling_mode == 'strict_half':
            samples = {}
            orig_sample = self.dataset.__getitem__(idx)

            new_idx = self.idx_sampling[idx]
            self.cnt_samples += 1
            if self.cnt_samples >= self.num_samples:
                self.init_counts()

            new_sample = self.dataset.__getitem__(new_idx)

            samples.update(orig_sample)
            for k, v in new_sample.items():
                samples['os_' + k] = v

            return samples
