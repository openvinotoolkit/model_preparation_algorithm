#__norm_cfg = dict(type='SyncBN', requires_grad=True)
__norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='ClassIncrSegmentor',
    is_task_adapt=True,
    num_stages=2,
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
                enable=False
            ),
            add_input=False
        )
    ),
    decode_head=[
        dict(type='FCNHead',
             in_channels=[40, 80, 160, 320],
             in_index=[0, 1, 2, 3],
             input_transform='multiple_select',
             channels=40,
             kernel_size=1,
             num_convs=0,
             concat_input=False,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=__norm_cfg,
             align_corners=False,
             enable_aggregator=True,
             enable_out_norm=False,
             loss_decode=[
                 dict(type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_jitter_prob=0.01,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
        dict(type='OCRHead',
             in_channels=[40, 80, 160, 320],
             in_index=[0, 1, 2, 3],
             input_transform='multiple_select',
             channels=40,
             ocr_channels=40,
             sep_conv=True,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=__norm_cfg,
             align_corners=False,
             enable_aggregator=True,
             enable_out_norm=True,
             loss_decode=[
                 dict(type='AMSoftmaxLoss',
                      scale_cfg=dict(
                          type='PolyScalarScheduler',
                          start_scale=30,
                          end_scale=5,
                          by_epoch=True,
                          num_iters=500,
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
                          num_iters=400,
                          power=1.2
                      ),
                      loss_jitter_prob=0.01,
                      border_reweighting=False,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
    ],
    train_cfg=dict(
        mix_loss=dict(
            enable=False,
            weight=0.1
        ),
        loss_reweighting=dict(
            weights={'decode_0.loss_seg': 0.9,
                     'decode_1.loss_seg': 1.0},
            momentum=0.1
        ),
    ),
    test_cfg=dict(
        mode='whole',
        output_scale=10.0,
    ),
)

find_unused_parameters = False

# optimizer
optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=1e-3,
    eps=1e-08,
    weight_decay=0.0
)

# parameter manager
params_config = dict(
    iters=0,
)
