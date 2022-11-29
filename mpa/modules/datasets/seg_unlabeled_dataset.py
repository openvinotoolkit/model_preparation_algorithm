from mmseg.datasets import DATASETS, build_dataset
import numpy as np
from mmseg.datasets import CustomDataset
from mmseg.datasets.pipelines import Compose

@DATASETS.register_module()
class UnlabeledSegDataset(CustomDataset):
    """Dataset wrapper for Semi-SL Semantic Seg experiments.
    Input : splits of labeled & unlabeld datasets
    """
    def __init__(self, orig_type=None, **kwargs):
        # Original dataset
        dataset_cfg = kwargs.copy()
        if 'cutmix' in dataset_cfg:
            self.cutmix_flag = dataset_cfg.pop('cutmix', False)
        else:
            self.cutmix_flag = False
        dataset_cfg['type'] = orig_type

        self.unlabeled_dataset = build_dataset(dataset_cfg)

        # Subsets
        self.num_unlabeled = len(self.unlabeled_dataset)
        self.unlabeled_index = np.random.permutation(self.num_unlabeled)
        if self.cutmix_flag:
            self.cutmix_unlabeled_index = np.random.permutation(self.num_unlabeled)
        print('----------- #Unlabeled: ', self.num_unlabeled)

    def __len__(self):
        """Total number of samples of data."""
        return self.num_unlabeled

    def __getitem__(self, idx):
        data = {}
        unlabeled_idx = int(self.unlabeled_index[idx])
        unlabeled_data = self.unlabeled_dataset[unlabeled_idx]

        return unlabeled_data
