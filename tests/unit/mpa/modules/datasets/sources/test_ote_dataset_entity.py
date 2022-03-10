import pytest
# from unittest.mock import patch, Mock

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

from ote_sdk.entities.datasets import Subset

from mpa.modules.datasets.sources.ote_dataset_entity import OTEDatasetEntity
from mpa.utils.logger import get_logger

logger = get_logger()


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_ote_dataset_entity_init(fixture_dataset_entity):
    source1 = OTEDatasetEntity(fixture_dataset_entity, Subset.TRAINING, label_list=['0', '1', '2'])
    assert source1.get_length() == 12  # .png image counts in tests/assets/dirs/classification/train

    source2 = OTEDatasetEntity(fixture_dataset_entity, Subset.VALIDATION, label_list=['0', '1', '2'])
    assert source2.get_length() == 12  # .png image counts in tests/assets/dirs/classification/val

    source3 = OTEDatasetEntity(fixture_dataset_entity, Subset.UNLABELED, label_list=['0', '1', '2'])
    assert source3.get_length() == 4  # .png image counts in tests/assets/dirs/classification/unlabeled

    for idx in range(source1.get_length()):
        sample = source1.get_sample(idx)
        logger.info(f"[{idx}] annotations: rectangles = {sample['rectangles']}, labels = {sample['labels']}")
        assert 'rectangles' in sample.keys()
        assert 'labels' in sample.keys()
        assert 'img_info' in sample.keys()
        assert 'img' in sample.keys()
