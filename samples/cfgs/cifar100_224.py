# dataset settings
img_norm_cfg = dict(
    mean=[129.3, 124.1, 112.4], std=[68.2,  65.4,  70.4], to_rgb=True)

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
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='TVDatasetSplit',
        base='CIFAR100',
        train=True,
        data_prefix='data/torchvision/cifar100',
        pipeline=train_pipeline,
        samples_per_gpu=16,
        workers_per_gpu=4,
        download=True
    ),
    val=dict(
        type='CIFAR100',
        train=False,
        data_prefix='data/torchvision/cifar100',
        pipeline=test_pipeline,
        test_mode=True,
        download=True
    ),
    test=dict(
        type='CIFAR100',
        train=False,
        data_prefix='data/torchvision/cifar100',
        pipeline=test_pipeline,
        test_mode=True,
        download=True
    )
)