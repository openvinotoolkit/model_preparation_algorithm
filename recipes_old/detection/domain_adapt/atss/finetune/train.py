_base_ = '../train.py'

name = 'det-da-atss-finetune-train'

model= dict(
    type='CustomATSS',
    super_type='UnbiasedTeacher',
    unlabeled_loss_weight=0.0,
    bbox_head=dict(
        type='CustomATSSHead',
        use_qfl=False,
        qfl_cfg=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
    ),
    l2sp_weight=0.0,  # 0.0001 to enable L2SP optimizer
)

#
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        type='PseudoSemiDataset',
        pseudo_length=2000,  # (2000/2=1000 iter per epoch)
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3
            ),
            dict(type='Resize',
                img_scale=[
                    ( 992, 736),
                    ( 896, 736),
                    (1088, 736),
                    ( 992, 672),
                    ( 992, 800),
                ],
                multiscale_mode='value',
                keep_ratio=False,
            ),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='BranchImage', key_map=dict(img='img0')),
            dict(type='NDArrayToPILImage', keys=['img']),
            dict(type='RandomApply',
                transform_cfgs=[dict(type='ColorJitter',
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )],
                p=0.8,
            ),
            dict(type='RandomGrayscale', p=0.2),
            dict(type='RandomApply',
                transform_cfgs=[dict(type='RandomGaussianBlur',
                    sigma_min=0.1,
                    sigma_max=2.0,
                )],
                p=0.5,
            ),
            dict(type='PILImageToNDArray', keys=['img']),
            dict(type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True,
            ),
            dict(type='Pad', size_divisor=32),
            dict(type='NDArrayToTensor', keys=['img', 'img0']),
            dict(type='RandomErasing',
                p=0.7,
                scale=[0.05, 0.2],
                ratio=[0.3, 3.3],
                value='random',
            ),
            dict(type='RandomErasing',
                p=0.5,
                scale=[0.02, 0.2],
                ratio=[0.10, 6.0],
                value='random',
            ),
            dict(type='RandomErasing',
                p=0.3,
                scale=[0.02, 0.2],
                ratio=[0.05, 8.0],
                value='random',
            ),
            dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            dict(type='ToDataContainer',
                fields=[
                    dict(key='img', stack=True),
                    dict(key='img0', stack=True),
                    dict(key='gt_bboxes'),
                    dict(key='gt_labels'),
                ],
            ),
            dict(type='Collect',
                keys=['img', 'img0', 'gt_bboxes', 'gt_labels',],
            ),
        ],
    ),
)

#total_epochs = 10
#lr_config = dict(
#    _delete_=True,
#    policy='step',
#    step=[8, 9],
#    warmup='linear',
#    warmup_iters=1000,
#    warmup_ratio=0.001,
#)
#runner = dict(type='EpochBasedRunner',
#    _delete_=True,
#)
workflow = [('train', 1)]

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
)
evaluation = dict(
    interval=1,
)

custom_hooks = [
    #dict(type='NumClassCheckHook'),
    dict(type='DualModelEMAHook',
        momentum=0.0004,
        start_epoch=2,
    ),
    dict(type='EarlyStoppingHook',
        patience=5,
        iteration_patience=1000,
        metric='bbox_mAP',
        interval=1,
        priority=75
    ),
]

optimizer_config = dict(
    type='SAMOptimizerHook',
    start_epoch=1000,  # set 1 ~ total_epochs to enable
    grad_clip=dict(
        max_norm=35,
        norm_type=2
    )
)
