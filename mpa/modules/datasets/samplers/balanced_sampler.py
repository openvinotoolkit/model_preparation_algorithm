import numpy as np
from torch.utils.data.sampler import Sampler
import math
from mpa.utils.logger import get_logger

logger = get_logger()


class BalancedSampler(Sampler):
    """Sampler for Class-Incremental Task
    This sampler is a sampler that creates an effective batch
    In reduce mode,
    reduce the iteration size by estimating the trials
    that all samples in the tail class are selected more than once with probability 0.999

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        efficient_mode (bool): Flag about using efficient mode
    """
    def __init__(self, dataset, batch_size, efficient_mode=True):
        self.batch_size = batch_size
        self.repeat = 1
        if hasattr(dataset, 'times'):
            self.repeat = dataset.times
        if hasattr(dataset, 'dataset'):
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset
        self.img_indices = self.dataset.img_indices
        self.num_cls = len(self.img_indices.keys())
        self.data_length = len(self.dataset)

        if efficient_mode:
            # Reduce the # of sampling (sampling data for a single epoch)
            self.num_tail = min([len(cls_indices) for cls_indices in self.img_indices.values()])
            base = 1 - (1/self.num_tail)
            if base == 0:
                raise ValueError('Required more than one sample per class')
            self.num_trials = int(math.log(0.001, base))
            if int(self.data_length / self.num_cls) < self.num_trials:
                self.num_trials = int(self.data_length / self.num_cls)
        else:
            self.num_trials = int(self.data_length / self.num_cls)
        self.compute_sampler_length()
        logger.info(f"This sampler will select balanced samples {self.num_trials} times")

    def compute_sampler_length(self):
        self.sampler_length = self.num_trials * self.num_cls * self.repeat

    def __iter__(self):
        indices = []
        for _ in range(self.repeat):
            for i in range(self.num_trials):
                indice = np.concatenate(
                    [np.random.choice(self.img_indices[cls_indices], 1) for cls_indices in self.img_indices.keys()])
                indices.append(indice)

        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()

        return iter(indices)

    def __len__(self):
        return self.sampler_length
