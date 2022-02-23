import pytest
from mmdet.datasets import build_dataset
from mpa.modules.datasets.det_incr_dataset import DetIncrCocoDataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_sampling_incr_coco_dataset():

    data_cfg = dict(
        type='DetIncrCocoDataset',
        img_ids_dict=dict(
            img_ids=[],
            img_ids_old=[],
            img_ids_new=[],
            old_classes=['person'],
            new_classes=['car']
        ),
        org_type='CocoDataset',
        classes=['car'],
        ann_file='tests/assets/data_cfg/instances_train2017.1@0.01.json',
        pipeline=[],
    )
    dataset = build_dataset(data_cfg)
    assert isinstance(dataset, DetIncrCocoDataset)
