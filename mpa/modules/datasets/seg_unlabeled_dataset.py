from mmseg.datasets import DATASETS, build_dataset
import numpy as np
from mpa.utils.logger import get_logger

from mmseg.datasets import CustomDataset

logger = get_logger()


@DATASETS.register_module()
class UnlabeledSegDataset(CustomDataset):
    """Dataset wrapper for Semi-SL Semantic Seg experiments.
    Input : splits of labeled & unlabeld datasets
    """
    def __init__(self, orig_type=None, **kwargs):
        # Original dataset
        dataset_cfg = kwargs.copy()
        if 'cutmix' in dataset_cfg:
            logger.warning("Currently cutmix is not supported. It will be added soon.")
        dataset_cfg['type'] = orig_type

        self.unlabeled_dataset = build_dataset(dataset_cfg)

        # Subsets
        self.num_unlabeled = len(self.unlabeled_dataset)
        self.unlabeled_index = np.random.permutation(self.num_unlabeled)

    def __len__(self):
        """Total number of samples of data."""
        return self.num_unlabeled

    def __getitem__(self, idx):
        unlabeled_idx = int(self.unlabeled_index[idx])
        unlabeled_data = self.unlabeled_dataset[unlabeled_idx]

        return unlabeled_data
