import pytest
import torch
from mpa.modules.models.detectors.custom_single_stage_detector import CustomSingleStageDetector

from mmdet.models.builder import build_detector

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_custom_single_stage_detector_build():
    image_width, image_height = (512, 512)
    model_cfg = dict(
        type='CustomSingleStageDetector',
        backbone=dict(
            type='mobilenetv2_w1',
            out_indices=(4, 5),
            frozen_stages=-1,
            norm_eval=False,
            pretrained=True
        ),
        neck=None,
        bbox_head=dict(
            type='SSDHead',
            num_classes=1,
            in_channels=(int(96), int(320)),
            anchor_generator=dict(
                type='SSDAnchorGeneratorClustered',
                strides=(16, 32),
                widths=[
                    [image_width * x for x in
                     [0.015411783166343854, 0.033018232306549156, 0.04467156688464953,
                      0.0610697815328886]],
                    [image_width * x for x in
                     [0.0789599954420517, 0.10113984043326349, 0.12805187473050397, 0.16198319380154758,
                      0.21636496806213493]],

                ],
                heights=[
                    [image_height * x for x in
                     [0.05032631418898226, 0.10070800135152037, 0.15806180366055939,
                      0.22343401646383804]],
                    [image_height * x for x in
                     [0.300881401352503, 0.393181898580379, 0.4998807213337051, 0.6386035764261081,
                      0.8363451552091518]],

                ],
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2), ),
        ))

    model = build_detector(model_cfg)
    assert isinstance(model, CustomSingleStageDetector)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_custom_single_stage_detector_load_state_dict_pre_hook():
    chkpt_classes = ['person', 'car']
    model_classes = ['tree', 'car', 'person']
    chkpt_dict = {
        'bbox_head.cls_convs.0.weight': torch.tensor([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 0, 0, 0],  # BG
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 0, 0, 0],  # BG
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 0, 0, 0],  # BG
        ]),
        'bbox_head.cls_convs.0.bias': torch.tensor([
            [1],
            [2],
            [0],  # BG
            [1],
            [2],
            [0],  # BG
            [1],
            [2],
            [0],  # BG
        ]),
    }
    model_dict = {
        'bbox_head.cls_convs.0.weight': torch.tensor([
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [0, 0, 0, 0],  # BG
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [0, 0, 0, 0],  # BG
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [0, 0, 0, 0],  # BG
        ]),
        'bbox_head.cls_convs.0.bias': torch.tensor([
            [3],
            [4],
            [5],
            [0],  # BG
            [3],
            [4],
            [5],
            [0],  # BG
            [3],
            [4],
            [5],
            [0],  # BG
        ]),
    }
    gt_dict = {
        'bbox_head.cls_convs.0.weight': torch.tensor([
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [0, 0, 0, 0],  # BG
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [0, 0, 0, 0],  # BG
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [0, 0, 0, 0],  # BG
        ]),
        'bbox_head.cls_convs.0.bias': torch.tensor([
            [3],
            [2],
            [1],
            [0],  # BG
            [3],
            [2],
            [1],
            [0],  # BG
            [3],
            [2],
            [1],
            [0],  # BG
        ]),
    }
    class Model:
        def state_dict(self):
            return model_dict
    CustomSingleStageDetector.load_state_dict_pre_hook(
        Model(), model_classes, chkpt_classes, chkpt_dict, ''
    )
    for k, gt in gt_dict.items():
        assert (chkpt_dict[k] != gt).sum() == 0
