_base_ = [
    './pipelines/semisl_pipeline.py'
]

__train_pipeline = {{_base_.train_pipeline}}
__train_pipeline_strong = {{_base_.train_pipeline_strong}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    num_classes=10,
    train=dict(
        type='ClsTVDataset',
        base='SVHN',
        data_prefix='data/torchvision/svhn',
        split='train',
        num_images=100,
        pipeline=__train_pipeline,
        samples_per_gpu=16,
        workers_per_gpu=2,
        download=True,
    ),
    # Unlabeled Dataset
    unlabeled=dict(
        type='ClsTVDataset',
        base='SVHN',
        split='train',
        data_prefix='data/torchvision/svhn',
        num_images=400,
        pipeline=dict(
            weak=__train_pipeline,
            strong=__train_pipeline_strong
        ),
        samples_per_gpu=48,
        workers_per_gpu=2,
        download=True,
        use_labels=False
    ),
    val=dict(
        type='ClsTVDataset',
        base='SVHN',
        split='train',
        data_prefix='data/torchvision/svhn',
        num_images=1400,
        samples_per_gpu=64,
        workers_per_gpu=4,
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type='ClsTVDataset',
        base='SVHN',
        split='test',
        data_prefix='data/torchvision/svhn',
        num_images=-1,
        samples_per_gpu=128,
        workers_per_gpu=4,
        pipeline=__test_pipeline,
        download=True,
    )
)
