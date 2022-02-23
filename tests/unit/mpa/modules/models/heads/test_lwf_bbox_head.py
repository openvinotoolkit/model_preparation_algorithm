import pytest

from mmdet.models.builder import build_head

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

from mpa.modules.experimental.models.heads.lwf_bbox_head import OffLwfSSDHead


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_off_lwf_bbox_head():
    image_width, image_height = 512, 512
    org_model_classes = ['person']
    model_classes = ['person', 'car']
    bbox_head = dict(
        type='OffLwfSSDHead',
        num_classes=2,
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
        loss_cls_lwf=dict(
            type='LwFLoss',
            src_classes=org_model_classes,
            dst_classes=model_classes,
            bg_aware=True,
            temperature=2.0,
            loss_weight=1.0
        ),
        src_classes=org_model_classes,
        dst_classes=model_classes
    )
    head = build_head(bbox_head)
    assert isinstance(head, OffLwfSSDHead)
