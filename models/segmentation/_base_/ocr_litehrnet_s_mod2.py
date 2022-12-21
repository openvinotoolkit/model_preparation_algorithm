__norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='ClassIncrSegmentor',
    pretrained=None,
    backbone=dict(
        type='LiteHRNet',
        norm_cfg=__norm_cfg,
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1,
                strides=(2, 2),
                extra_stride=True,
                input_norm=False,
            ),
            num_stages=2,
            stages_spec=dict(
                neighbour_weighting=False,
                weighting_module_version='v1',
                num_modules=(4, 4),
                num_branches=(2, 3),
                num_blocks=(2, 2),
                module_type=('LITE', 'LITE'),
                with_fuse=(True, True),
                reduce_ratios=(8, 8),
                num_channels=(
                    (60, 120),
                    (60, 120, 240),
                )
            ),
            out_modules=dict(
                conv=dict(
                    enable=False,
                    channels=160
                ),
                position_att=dict(
                    enable=False,
                    key_channels=64,
                    value_channels=240,
                    psp_size=(1, 3, 6, 8),
                ),
                local_att=dict(
                    enable=False
                )
            ),
            out_aggregator=dict(
                enable=False
            ),
            add_input=False
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=[60, 120, 240],
        in_index=[0, 1, 2],
        input_transform='multiple_select',
        channels=60,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=__norm_cfg,
        align_corners=False,
        enable_aggregator=True,
        aggregator_merge_norm=None,
        aggregator_use_concat=False,
        enable_out_norm=False,
        enable_loss_equalizer=True,
        loss_decode=[
            dict(type='CrossEntropyLoss',
                 use_sigmoid=False,
                 sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                 loss_weight=1.0),
        ]),
)
