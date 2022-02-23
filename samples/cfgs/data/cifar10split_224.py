_base_ = [
    './pipelines/rcrop_hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        RandomCrop=dict(
            size=32,
            padding=4,
            padding_mode='reflect'
        ),
        Normalize=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
    ),
    train=dict(
            type='TVDatasetSplit',
            base='CIFAR10',
            train=True,
            data_prefix='data/torchvision/cifar10',
            pipeline=__train_pipeline,
            samples_per_gpu=16,
            workers_per_gpu=4,
            download=True
    ),
    val=dict(
        type='CIFAR10',
        train=False,
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline,
        test_mode=True
    ),
    test=dict(
        type='CIFAR10',
        train=False,
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline,
        test_mode=True
    )
)
