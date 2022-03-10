from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.polygon import Polygon
from ote_sdk.entities.shapes.ellipse import Ellipse

from mpa.modules.datasets.sources import DATASOURCES
from mpa.utils.logger import get_logger

logger = get_logger()


@DATASOURCES.register_module()
class OTEDatasetEntity(object):
    def __init__(self, source: DatasetEntity, split: Subset, label_list: list, **kwargs):
        self.labels = label_list
        self._prepare(source, split)

    def _prepare(self, source: DatasetEntity, split: Subset):
        logger.info(f'_prepare() total len = {len(source)}')
        self.source = source.get_subset(split)
        logger.info(f'for subset {split}, len = {len(self.source)}')

    def get_length(self):
        return len(self.source)

    def get_sample(self, idx, roi: Annotation = None):
        dataset_item = self.source[idx]
        logger.info(f'sampled datasetitem entity = {dataset_item}')
        sample = dict()
        if roi is not None:
            sample['img'] = dataset_item.roi_numpy(roi=roi)
        else:
            sample['img'] = dataset_item.numpy
        sample['img_info'] = dict()
        sample['img_info']['width'] = dataset_item.width
        sample['img_info']['height'] = dataset_item.height
        if len(dataset_item.annotation_scene.annotations):
            for annotation in dataset_item.annotation_scene.annotations:
                if type(annotation.shape) == Rectangle:
                    shape = 'rectangles'
                elif type(annotation.shape) == Polygon:
                    shape = 'points'
                elif type(annotation.shape) == Ellipse:
                    shape = 'ellipses'
                else:
                    raise ValueError(f'undefined ShapeType {type(annotation.shape)}')

                if shape not in sample:
                    sample[shape] = []
                sample[shape].append(annotation.shape)
                if 'labels' not in sample:
                    sample['labels'] = []
                sample['labels'].extend(annotation.get_labels())

        return sample
