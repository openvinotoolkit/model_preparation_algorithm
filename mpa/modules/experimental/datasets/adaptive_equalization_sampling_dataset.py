import torch
import torch.multiprocessing as mp

from mmseg.datasets import DATASETS, build_dataset


@DATASETS.register_module()
class AdaptiveEqualizationSamplingDataset(object):
    """Dataset wrapper for Adaptive Equalization Sampling.
    """

    def __init__(self,
                 orig_type=None,
                 **kwargs):

        dataset_cfg = kwargs.copy()
        dataset_cfg['type'] = orig_type
        self.dataset = build_dataset(dataset_cfg)
        self.num_samples = len(self.dataset)

        # why is it used with (3, class) shape?
        self._class_criterion = torch.rand(len(self.dataset.CLASSES)).type(torch.float32)
        print('####################### AdaptiveEqualizationSamplingDataset')

    def update_confidence_bank(self, conf):
        print(id(self), 'before :', self._class_criterion)
        self._class_criterion = conf
        # dist.broadcast(self._class_criterion, 0)
        print(id(self), 'after :', self._class_criterion)
        print(dir(mp))
        print(mp.active_children())

        ctx = mp.get_context('spawn')
        print(ctx)
        for children in mp.active_children():
            print(children)
            print(dir(children))

    @property
    def class_criterion(self):
        return self._class_criterion

    def __len__(self):
        """Total number of samples of data."""
        return self.num_samples

    def __getitem__(self, idx):
        # sample_from_bank
        worker_info = torch.utils.data.get_worker_info()

        print(worker_info, 'self._class_criterion :', self._class_criterion)
        print(worker_info.dataset.class_criterion)
        return self.dataset.__getitem__(idx)
