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
        
        pipeline = dataset_cfg["pipeline"]
        self.strong_pipeline = Compose(pipeline.get('strong_aug', None))
        dataset_cfg["pipeline"] = pipeline.get('weak_aug', None)

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
        unlabeled_raw_item = self.unlabeled_dataset._get_raw_unlabeled_data(unlabeled_idx)
        unlabeled_weak = self.unlabeled_dataset.pipeline(unlabeled_raw_item)
        unlabeled_strong = self.strong_pipeline(unlabeled_raw_item)
        
        for k, v in unlabeled_weak.items():
            data['ul_w_' + k] = v
        for k, v in unlabeled_strong.items():
            data['ul_s_' + k] = v

        if self.cutmix_flag:
            if self.num_unlabeled > 0:
                cutmix_idx = int(self.cutmix_unlabeled_index[idx])
                cutmix_data = self.unlabeled_dataset[cutmix_idx]
                for k, v in cutmix_data.items():
                    data['ul2_' + k] = v
        return data
