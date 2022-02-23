import pytest

from mpa.modules.datasets.task_adapt_dataset_seg import TaskAdaptDatasetSeg

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_build_task_adapt_dataset_seg():
    dataset = TaskAdaptDatasetSeg(
        model_classes=['background', 'car', 'building'],
        with_background=True,
        org_type='CityscapesDataset',
        classes=['building'],
        img_dir='tests/assets/dirs/segmentation/train',
        pipeline=[]
    )

    assert len(dataset) == len(dataset.dataset)
    assert dataset.model_classes == dataset.dataset.CLASSES

    dataset_repeat = TaskAdaptDatasetSeg(
        model_classes=['background', 'car', 'building'],
        with_background=True,
        org_type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CityscapesDataset',
            classes=['building'],
            img_dir='tests/assets/dirs/segmentation/train',
            pipeline=[]
        )
    )

    assert len(dataset_repeat) == len(dataset_repeat.dataset)
    assert len(dataset_repeat.dataset) == len(dataset_repeat.dataset.dataset)
    assert dataset_repeat.model_classes == dataset_repeat.dataset.CLASSES
    assert dataset_repeat.dataset.CLASSES == dataset_repeat.dataset.dataset.CLASSES
