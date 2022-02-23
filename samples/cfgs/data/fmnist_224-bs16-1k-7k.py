_base_ = ["./pipelines/semisl_pipeline.py"]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}
__train_pipeline_strong = {{_base_.train_pipeline_strong}}

seed = 1234

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    num_classes=10,
    train=[
        dict(
            type='TVDatasetSplit',
            base='FashionMNIST',
            num_classes=10,
            data_prefix='data/torchvision/fmnist',
            train=True,
            num_images=1000,
            pipeline=__train_pipeline,
            samples_per_gpu=16,
            workers_per_gpu=2,
            seed=seed,
            download=True,
        ),
        dict(
            type='TVDatasetSplit',
            base='FashionMNIST',
            train=True,
            num_classes=10,
            data_prefix='data/torchvision/fmnist',
            num_images=7000,
            pipeline=dict(
                weak=__train_pipeline,
                strong=__train_pipeline_strong
            ),
            samples_per_gpu=112,
            workers_per_gpu=2,
            seed=seed,
            download=True,
            use_labels=False
        )
    ],
    val=dict(
        type="TVDatasetSplit",
        base="FashionMNIST",
        train=True,
        data_prefix="data/torchvision/fmnist",
        num_images=12000,
        samples_per_gpu=128,
        workers_per_gpu=4,
        seed=seed,
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type="TVDatasetSplit",
        base="FashionMNIST",
        train=False,
        data_prefix="data/torchvision/fmnist",
        num_images=-1,
        samples_per_gpu=128,
        workers_per_gpu=4,
        seed=seed,
        pipeline=__test_pipeline,
        download=True,
    ),
)
