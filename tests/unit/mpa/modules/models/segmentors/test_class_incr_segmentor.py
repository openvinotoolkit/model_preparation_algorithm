import pytest
import torch
from mpa.modules.models.segmentors.class_incr_segmentor import ClassIncrSegmentor

from mmseg.models.builder import build_segmentor

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_class_incr_segmentor_build():
    model_cfg = dict(
        type='ClassIncrSegmentor',
        pretrained=None,
        is_task_adapt=True,
        task_adapt=dict(
            src_classes=['background', 'car'],
            dst_classes=['background', 'car', 'building']),
        num_stages=2,
        backbone=dict(
            type='LiteHRNet',
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            extra=dict(
                stem=dict(
                    stem_channels=32,
                    out_channels=32,
                    expand_ratio=1,
                    strides=(2, 2),
                    extra_stride=False,
                    input_norm=False,
                ),
                num_stages=3,
                stages_spec=dict(
                    num_modules=(2, 4, 2),
                    num_branches=(2, 3, 4),
                    num_blocks=(2, 2, 2),
                    module_type=('LITE', 'LITE', 'LITE'),
                    with_fuse=(True, True, True),
                    reduce_ratios=(8, 8, 8),
                    num_channels=(
                        (40, 80),
                        (40, 80, 160),
                        (40, 80, 160, 320),
                    )
                ),
                out_modules=dict(
                    conv=dict(
                        enable=False,
                        channels=320
                    ),
                    position_att=dict(
                        enable=False,
                        key_channels=128,
                        value_channels=320,
                        psp_size=(1, 3, 6, 8),
                    ),
                    local_att=dict(
                        enable=False
                    )
                ),
                out_aggregator=dict(
                    enable=True
                ),
                add_input=False
            )
        ),
        decode_head=[
            dict(type='FCNHead',
                 in_channels=40,
                 in_index=0,
                 channels=40,
                 input_transform=None,
                 kernel_size=1,
                 num_convs=0,
                 concat_input=False,
                 dropout_ratio=-1,
                 num_classes=19,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 align_corners=False,
                 enable_out_norm=False,
                 loss_decode=[
                    dict(type='RecallLoss',
                         loss_jitter_prob=0.01,
                         sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
                         loss_weight=1.0)
                 ]),
            dict(type='OCRHead',
                 in_channels=40,
                 in_index=0,
                 channels=40,
                 ocr_channels=40,
                 sep_conv=True,
                 input_transform=None,
                 dropout_ratio=-1,
                 num_classes=19,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 align_corners=False,
                 enable_out_norm=True,
                 loss_decode=[
                    dict(type='AMSoftmaxLoss',
                         scale_cfg=dict(
                             type='PolyScalarScheduler',
                             start_scale=30,
                             end_scale=5,
                             by_epoch=True,
                             num_iters=250,
                             power=1.2
                         ),
                         margin_type='cos',
                         margin=0.5,
                         gamma=0.0,
                         t=1.0,
                         target_loss='ce',
                         pr_product=False,
                         conf_penalty_weight=dict(
                             type='PolyScalarScheduler',
                             start_scale=0.2,
                             end_scale=0.15,
                             by_epoch=True,
                             num_iters=200,
                             power=1.2
                         ),
                         loss_jitter_prob=0.01,
                         border_reweighting=False,
                         sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                         loss_weight=1.0)
                         ])
        ]
    )

    model = build_segmentor(model_cfg)
    model_cfg.update({'is_task_adapt': False})
    model_no_TA = build_segmentor(model_cfg)

    assert isinstance(model, ClassIncrSegmentor)
    assert isinstance(model_no_TA, ClassIncrSegmentor)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_class_incr_segmentor_load_state_dict_pre_hook():
    chkpt_classes = ['person', 'car']
    model_classes = ['tree', 'car', 'person']

    chkpt_dict = {
        'decode_head.0.conv_seg.weight': torch.tensor([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]),
        'decode_head.0.conv_seg.bias': torch.tensor([
            [1],
            [2],
        ]),
    }
    model_dict = {
        'decode_head.0.conv_seg.weight': torch.tensor([
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
        ]),
        'decode_head.0.conv_seg.bias': torch.tensor([
            [3],
            [4],
            [5],
        ]),
    }
    gt_dict = {
        'decode_head.0.conv_seg.weight': torch.tensor([
            [3, 3, 3, 3],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
        ]),
        'decode_head.0.conv_seg.bias': torch.tensor([
            [3],
            [2],
            [1],
        ]),
    }

    class Model:
        def state_dict(self):
            return model_dict

    ClassIncrSegmentor.load_state_dict_pre_hook(
        Model(), model_classes, chkpt_classes, chkpt_dict, ''
    )

    for k, gt in gt_dict.items():
        assert (chkpt_dict[k] != gt).sum() == 0
