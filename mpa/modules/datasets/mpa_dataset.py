from mmdet.datasets.builder import DATASETS
from mmdet.apis.ote.extension.datasets import OTEDataset
from mpa.utils.logger import get_logger

logger = get_logger()


@DATASETS.register_module()
class MPADataset(OTEDataset):
    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.get_ann_info(idx)['labels'].astype(int).tolist()
