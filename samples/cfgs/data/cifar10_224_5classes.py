_base_ = [
    './pipelines/hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}
__dataset_type = 'TVDatasetSplit'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    num_classes=5,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        Normalize=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
    ),
    train=dict(
        type=__dataset_type,
        base='CIFAR10',
        data_prefix='./data',
        pipeline=__train_pipeline,
        classes=[0, 1, 2, 3, 4]

    ),
    test=dict(
        type=__dataset_type,
        base='CIFAR10',
        data_prefix='./data',
        pipeline=__test_pipeline,
        test_mode=True,
        classes=[0, 1, 2, 3, 4]
    ),
    val=dict(
        type=__dataset_type,
        base='CIFAR10',
        data_prefix='./data',
        pipeline=__test_pipeline,
        test_mode=True,
        classes=[0, 1, 2, 3, 4]
    )
)
