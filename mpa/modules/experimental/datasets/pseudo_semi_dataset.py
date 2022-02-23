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
        unlabeled_split = dataset_cfg.pop('unlabeled_split', '')
        if unlabeled_split == '':
            raise ValueError('You should use unlabeled data and put the path @ unlabeled_ann_dir!')
        if 'cutmix' in dataset_cfg:
            self.cutmix_flag = dataset_cfg.pop('cutmix', False)
        else:
            self.cutmix_flag = False

        dataset_cfg['type'] = orig_type
        self.labeled_dataset = build_dataset(dataset_cfg)
        self.CLASSES = self.labeled_dataset.CLASSES
        self.PALETTE = self.labeled_dataset.PALETTE

        dataset_cfg['split'] = unlabeled_split
        self.unlabeled_dataset = build_dataset(dataset_cfg)

        # Subsets
        self.num_labeled = len(self.labeled_dataset)
        self.num_unlabeled = len(self.unlabeled_dataset)
        self.labeled_index = np.random.permutation(max(self.num_labeled, self.num_unlabeled))
        self.unlabeled_index = np.random.permutation(max(self.num_labeled, self.num_unlabeled))
        if self.cutmix_flag:
            self.unlabeled_index2 = np.random.permutation(max(self.num_labeled, self.num_unlabeled))
        print('----------- #Labeled: ', self.num_labeled)
        print('----------- #Unlabeled: ', self.num_unlabeled)

    def __len__(self):
        """Total number of samples of data."""
        return self.num_labeled

    def __getitem__(self, idx):
        data = {}
        labeled_idx = self.labeled_index[idx] % self.num_labeled
        labeled_data = self.labeled_dataset[labeled_idx]
        data.update(labeled_data)
        if self.num_unlabeled > 0:
            tmp = self.num_unlabeled / self.num_labeled
        if tmp > 0:
            idx = (idx + np.random.randint(tmp)*self.num_labeled) % self.num_unlabeled
            unlabeled_idx = self.unlabeled_index[idx] % self.num_unlabeled
            unlabeled_data = self.unlabeled_dataset[unlabeled_idx]
            for k, v in unlabeled_data.items():
                data['ul_' + k] = v
        if self.cutmix_flag:
            if self.num_unlabeled > 0:
                unlabeled_idx = self.unlabeled_index2[idx] % self.num_unlabeled
                unlabeled_data = self.unlabeled_dataset[unlabeled_idx]
                for k, v in unlabeled_data.items():
                    data['ul2_' + k] = v
        return data
