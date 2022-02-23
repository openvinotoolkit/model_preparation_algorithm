# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomCrop', size=96, padding=int(96*0.125), padding_mode='reflect'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
train_pipeline_strong = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomCrop', size=96, padding=int(96*0.125), padding_mode='reflect'),
    dict(type='RandAugment', n=2, m=10),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=[
        # labeled
        dict(
            type='TVDatasetSplit',
            base='STL10',
            data_prefix='data/torchvision/stl10',
            split='train',
            num_images=500,
            pipeline=train_pipeline,
            samples_per_gpu=8,
            workers_per_gpu=4,
            download=True,
        ),
        # unlabeled
        dict(
            type='TVDatasetSplit',
            base='STL10',
            split='unlabeled',
            data_prefix='data/torchvision/stl10',
            num_images=3500,
            pipeline=dict(
                weak=train_pipeline,
                strong=train_pipeline_strong
            ),
            samples_per_gpu=56,
            workers_per_gpu=4,
            download=True,
            use_labels=False
        )
    ],
    val=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='test',
        data_prefix='data/torchvision/stl10',
        num_images=1000,
        pipeline=test_pipeline,
        download=True,
    ),
    test=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='test',
        data_prefix='data/torchvision/stl10',
        num_images=2000,
        pipeline=test_pipeline,
        download=True,
    )
)
