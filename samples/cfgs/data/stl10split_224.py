_base_ = [
    './pipelines/hflip_resize.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    num_classes=10,
    pipeline_options=dict(
        Resize=dict(
            size=224
        ),
        RandomResizedCrop=dict(
            size=(112, 112)
        ),
        Normalize=dict(
            # mean=[123.675, 116.28, 103.53],
            # std=[58.395, 57.12, 57.375]
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ),
    train=dict(
            type='TVDatasetSplit',
            base='STL10',
            data_prefix='data/torchvision/stl10',
            split='train',
            pipeline=__train_pipeline,
            num_images=1000,
            download=True,
    ),
    val=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='test',
        data_prefix='data/torchvision/stl10',
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='test',
        data_prefix='data/torchvision/stl10',
        pipeline=__test_pipeline,
        download=True,
    ),
    unlabeled=dict(
        type='TVDatasetSplit',
        base='STL10',
        split='unlabeled',
        data_prefix='data/torchvision/stl10',
        num_images=1000,
        download=True,
        use_labels=False
    )
)
