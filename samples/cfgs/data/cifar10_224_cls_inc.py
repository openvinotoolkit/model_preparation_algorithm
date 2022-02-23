_base_ = [
    './pipelines/rcrop_hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    num_classes=6,
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
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=__train_pipeline,
        new_classes=[6],
        num_images=200,
    ),
    val=dict(
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline,
        test_mode=True,
        classes=[0, 1, 2, 3, 4, 6],
        num_images=200,
    ),
    test=dict(
        type='ClsTVDataset',
        base='CIFAR10',
        data_prefix='data/torchvision/cifar10',
        pipeline=__test_pipeline,
        test_mode=True,
        classes=[0, 1, 2, 3, 4, 6],
        num_images=200,
    )
)
