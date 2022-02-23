import pytest
import torch
from mpa.modules.models.detectors.custom_atss_detector import CustomATSS

from mmdet.models.builder import build_detector

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_custom_atss_build():
    model_cfg = dict(
        type='CustomATSS',
        backbone=dict(
            type='mobilenetv2_w1',
            out_indices=(2, 3, 4, 5),
            frozen_stages=-1,
            norm_eval=False,
            pretrained=True
        ),
        neck=dict(
            type='FPN',
            in_channels=[24, 32, 96, 320],
            out_channels=64,
            start_level=1,
            add_extra_convs=True,
            extra_convs_on_inputs=False,
            num_outs=5,
            relu_before_extra_convs=True
        ),
        bbox_head=dict(
            type='ATSSHead',
            num_classes=2,
            in_channels=64,
            stacked_convs=4,
            feat_channels=64,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0
            ),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
            )
        ),
        task_adapt=dict(
            src_classes = ['person', 'car'],
            dst_classes = ['tree', 'car', 'person'],
        ),
    )

    model = build_detector(model_cfg)
    assert isinstance(model, CustomATSS)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_custom_atss_load_state_dict_pre_hook():
    chkpt_classes = ['person', 'car']
    model_classes = ['tree', 'car', 'person']
    chkpt_dict = {
        'bbox_head.atss_cls.weight': torch.tensor([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]),
        'bbox_head.atss_cls.bias': torch.tensor([
            [1],
            [2],
        ]),
    }
    model_dict = {
        'bbox_head.atss_cls.weight': torch.tensor([
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
        ]),
        'bbox_head.atss_cls.bias': torch.tensor([
            [3],
            [4],
            [5],
        ]),
    }
    gt_dict = {
        'bbox_head.atss_cls.weight': torch.tensor([
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
        ]),
        'bbox_head.atss_cls.bias': torch.tensor([
            [3],
            [2],
            [1],
        ]),
    }
    class Model:
        def state_dict(self):
            return model_dict
    CustomATSS.load_state_dict_pre_hook(
        Model(), model_classes, chkpt_classes, chkpt_dict, ''
    )
    for k, gt in gt_dict.items():
        assert (chkpt_dict[k] != gt).sum() == 0
