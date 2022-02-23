# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomCrop', size=32, padding=int(32*0.125), padding_mode='reflect'),
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
    samples_per_gpu=32,
    workers_per_gpu=4,
    num_classes=6,
    train=dict(
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=train_pipeline,
        new_classes=[6],
        num_images=100
    ),
    train_old=dict(
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=train_pipeline,
        classes=[0, 1, 2, 3, 4],
    ),
    val=dict(
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=test_pipeline,
        classes=[0, 1, 2, 3, 4, 6]
    ),
    test=dict(
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=test_pipeline,
        classes=[0, 1, 2, 3, 4, 6]
    )
)
