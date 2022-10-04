__dataset_type = 'CocoDataset'
__data_root = 'data/coco/'

img_size = (992, 736)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

__train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='ProbCompose',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1
            )
        ], probs=[0.8]),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', kernel_size=23),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

__test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

__samples_per_gpu = 8
data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_train2017.json',
        img_prefix=__data_root + 'train2017/',
        pipeline=__train_pipeline),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
        pipeline=__test_pipeline),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
        pipeline=__test_pipeline)
)
