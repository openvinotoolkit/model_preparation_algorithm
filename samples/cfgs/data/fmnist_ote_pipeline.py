_base_ = ["./pipelines/fixmatch_pipeline.py"]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

seed = 1234

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    num_classes=10,
    train=dict(
        type="TVDatasetSplit",
        base="FashionMNIST",
        data_prefix="data/torchvision/fmnist",
        train=True,
        num_images=1000,
        pipeline=__train_pipeline,
        samples_per_gpu=16,
        workers_per_gpu=4,
        seed=seed,
        download=True,
    ),
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
