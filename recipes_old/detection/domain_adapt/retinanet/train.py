_base_ = '../../../../external/mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco.py'

name = 'det-da-retinanet-train'
type = 'DetectionTrainer'

task_adapt = dict(
    op='REPLACE',
)

hparams = dict(
    dummy=0,
)

model = dict(
    pretrained=None,
    backbone=dict(  # Replacding R50 by OTE MV2
        _delete_=True,
        type='mobilenetv2_w1',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        norm_eval=True,  # False in OTE setting
        pretrained=True,
    ),
    neck=dict(
        in_channels=[24, 32, 96, 320],
        out_channels=64,
    ),
    bbox_head=dict(
        in_channels=64,
        feat_channels=64,
    ),
)

data = dict(
    samples_per_gpu=2,
    train=dict(
        _delete_=True,
        type='PseudoSemiDataset',
        labeled_percent=100.0,
        use_unlabeled=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(
                type='Resize',
                img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672),
                           (992, 800)],
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
    ),
    val=dict(
        samples_per_gpu=2,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(992, 736),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ]
            )
        ],
    ),
    test=dict(
        samples_per_gpu=2,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(992, 736),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ],
            ),
        ],
    ),
)

optimizer = dict(lr=0.001)

lr_config = dict(
    _delete_=True,
    policy='ReduceLROnPlateau',
    metric='bbox_mAP',
    patience=3,
    iteration_patience=600,
    interval=1,
    min_lr=1e-06,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333)
runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
total_epochs = 300

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
)

evaluation = dict(
    _delete_=True,
    interval=1,
    metric_items=['mAP'],
    save_best='bbox_mAP',
)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ],
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        patience=5,
        iteration_patience=1000,
        metric='bbox_mAP',
        interval=1,
        priority=75,
    ),
]

gpu_ids = range(0, 1)
load_from = None
work_dir = None
