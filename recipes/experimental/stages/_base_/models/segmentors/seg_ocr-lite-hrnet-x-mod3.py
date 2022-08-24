norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LiteHRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=60,
                out_channels=60,
                expand_ratio=1,
                strides=(2, 1),
                extra_stride=False,
                input_norm=False,
            ),
            num_stages=4,
            stages_spec=dict(
                weighting_module_version='v1',
                num_modules=(2, 4, 4, 2),
                num_branches=(2, 3, 4, 5),
                num_blocks=(2, 2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True, True),
                reduce_ratios=(2, 4, 8, 8),
                num_channels=(
                    (18, 60),
                    (18, 60, 80),
                    (18, 60, 80, 160),
                    (18, 60, 80, 160, 320),
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
                enable=False
            ),
            add_input=False
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=[18, 60, 80, 160, 320],
        in_index=[0, 1, 2, 3, 4],
        input_transform='multiple_select',
        channels=60,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        enable_aggregator=True,
        aggregator_min_channels=60,
        aggregator_merge_norm=None,
        aggregator_use_concat=False,
        enable_out_norm=False,
        enable_loss_equalizer=True,
        loss_decode=[
            dict(type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_jitter_prob=0.01,
                sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                loss_weight=4.0),
            dict(type='GeneralizedDiceLoss',
                smooth=1.0,
                gamma=5.0,
                alpha=0.5,
                beta=0.5,
                focal_gamma=1.0,
                loss_jitter_prob=0.01,
                loss_weight=4.0),
        ]
    ),
    train_cfg=dict(
        mix_loss=dict(
            enable=False,
            weight=0.1
        ),
    ),
    test_cfg=dict(
        mode='whole',
        output_scale=10.0,
    ),
)
