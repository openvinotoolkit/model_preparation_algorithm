# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from torch.utils.data.sampler import Sampler
from mpa.modules.utils.task_adapt import unwrap_dataset


class ClsIncrSampler(Sampler):
    """Sampler for Class-Incremental Task
    This sampler is a sampler that creates an effective batch
    For default setting,
    the square root of (number of old data/number of new data) is used as the ratio of old data
    In effective mode,
    the ratio of old and new data is used as 1:1

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        efficient_mode (bool): Flag about using efficient mode
    """
    def __init__(self, dataset, samples_per_gpu, efficient_mode=False):
        self.samples_per_gpu = samples_per_gpu

        # Dataset Wrapping remove & repeat for RepeatDataset
        self.dataset, self.repeat = unwrap_dataset(dataset)

        if hasattr(self.dataset, 'img_indices'):
            self.new_indices = self.dataset.img_indices['new']
            self.old_indices = self.dataset.img_indices['old']
        else:
            raise TypeError(f'{self.dataset} type does not have img_indices')

        if not len(self.new_indices) > 0:
            self.new_indices = self.old_indices
            self.old_indices = []

        old_new_ratio = np.sqrt(len(self.old_indices) / len(self.new_indices))

        if efficient_mode:
            self.data_length = int(len(self.new_indices) * (1 + old_new_ratio))
            self.old_new_ratio = 1
        else:
            self.data_length = len(self.dataset)
            self.old_new_ratio = int(old_new_ratio)

    def __iter__(self):
        indices = []
        for _ in range(self.repeat):
            for i in range(int(self.data_length / (1 + self.old_new_ratio))):
                indice = np.concatenate(
                    [np.random.choice(self.new_indices, 1),
                     np.random.choice(self.old_indices, self.old_new_ratio)])
                indices.append(indice)

        indices = np.concatenate(indices)
        num_extra = int(
            np.ceil(self.data_length * self.repeat / self.samples_per_gpu)
        ) * self.samples_per_gpu - len(indices)
        indices = np.concatenate(
            [indices, np.random.choice(indices, num_extra)])
        indices = indices.astype(np.int64).tolist()
        return iter(indices)

    def __len__(self):
        return self.data_length * self.repeat
