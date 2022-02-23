import pytest
import os
import numpy as np

from mmcv.parallel import DataContainer as DC

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

from mpa.modules.experimental.datasets.pseudo_incr_dataset import (
    PseudoIncrCocoDataset, CocoDataset, FormatPseudoLabels, LoadPseudoLabels, PseudoMinIoURandomCrop
)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_pseudo_incr_dataset():

    org_dataset = CocoDataset(
        ann_file='tests/assets/data_cfg/instances_train2017.1@0.01.json',
        pipeline=[],
    )

    new_anns = []
    for idx, org_ann in enumerate(org_dataset.data_infos):
        if idx % 2 == 0:
            new_anns.append([[]])
        else:
            new_anns.append([[np.array([
                0.0, 1.0, 2.0, 4.0, 1.0, 1.0, 0.0
            ])]])

    pseudo_labels = dict(
        detections=new_anns,
        classes=['NEW_CLASS']
    )

    np.save('tmp.npy', pseudo_labels, allow_pickle=True)

    dataset = PseudoIncrCocoDataset(
        ann_file='tests/assets/data_cfg/instances_train2017.1@0.01.json',
        pipeline=[],
        pre_stage_res='tmp.npy'
    )
    os.remove('tmp.npy')

    assert len(dataset) == len(org_dataset)
    assert len(dataset.CLASSES) == len(org_dataset.CLASSES)+1


@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_format_pseudo_labels():
    op = FormatPseudoLabels()
    data = dict(pseudo_labels=np.zeros(1))
    data = op(data)
    assert isinstance(data['pseudo_labels'], DC)


@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_load_pseudo_labels():
    op = LoadPseudoLabels()
    data = dict(ann_info=dict(pseudo_labels=np.zeros(1)))
    data = op(data)
    assert isinstance(data['pseudo_labels'], np.ndarray)


@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_pseudo_min_iou_random_crop():
    op = PseudoMinIoURandomCrop()
    dummy_bboxes = []
    for i in range(5):
        r = np.random.randint(224)
        dummy_bboxes.append(np.array([0., 0., 32., 32.])+r)
    dummy_gt_labels = np.random.randint(2, size=5)
    dummy_pseudo_labels = np.random.rand(5, 2)
    data = dict(img=np.random.rand(256, 256, 3),
                bbox_fields=['gt_bboxes_ignore', 'gt_bboxes'],
                gt_bboxes=np.array(dummy_bboxes),
                gt_bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
                gt_labels=dummy_gt_labels,
                pseudo_labels=dummy_pseudo_labels
                )
    res = op(data)
    assert len(res['pseudo_labels']) == len(res['gt_labels'])
