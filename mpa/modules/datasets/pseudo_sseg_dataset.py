from mmseg.datasets import DATASETS, build_dataset
import numpy as np

@DATASETS.register_module()
class PseudoSemanticSegDataset(object):
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
        #self.CLASSES = self.labeled_dataset.CLASSES
        #self.PALETTE = self.labeled_dataset.PALETTE

        self.unlabeled_dataset = build_dataset(dataset_cfg)

        # Subsets
        self.num_unlabeled = len(self.unlabeled_dataset)
        self.unlabeled_index = np.random.permutation(self.num_unlabeled)
        if self.cutmix_flag:
            self.unlabeled_index2 = np.random.permutation(self.num_unlabeled)
        print('----------- #Unlabeled: ', self.num_unlabeled)

    def __len__(self):
        """Total number of samples of data."""
        return self.num_unlabeled

    def __getitem__(self, idx):
        data = {}
        unlabeled_idx = int(self.unlabeled_index[idx])
        unlabeled_data = self.unlabeled_dataset[unlabeled_idx]
        for k, v in unlabeled_data.items():
            data['ul_' + k] = v
        if self.cutmix_flag:
            if self.num_unlabeled > 0:
                unlabeled_idx = int(self.unlabeled_index2[idx])
                unlabeled_data = self.unlabeled_dataset[unlabeled_idx]
                for k, v in unlabeled_data.items():
                    data['ul2_' + k] = v
        return data
