# data settings
_base_ = [
    '../../_base_/data.py'
]

# image norm meta data from the Imagenet dataset
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    num_classes=10,
    train=dict(
        type='CIFAR10',
        data_prefix='./data',
        pipeline=train_pipeline
    ),
    test=dict(
        type='CIFAR10',
        data_prefix='./data',
        pipeline=test_pipeline,
        test_mode=True
    ),
    val=dict(
        type='CIFAR10',
        data_prefix='./data',
        pipeline=test_pipeline,
        test_mode=True
    )
)
