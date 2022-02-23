_base_ = ["./pipelines/semisl_wo_hflip.py"]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

seed = 1234

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    num_classes=10,
    train=dict(
        type="TVDatasetSplit",
        base="SVHN",
        data_prefix="data/torchvision/svhn",
        split="train",
        num_images=1000,
        pipeline=__train_pipeline,
        samples_per_gpu=16,
        workers_per_gpu=4,
        seed=seed,
        download=True,
    ),
    val=dict(
        type="TVDatasetSplit",
        base="SVHN",
        split="train",
        data_prefix="data/torchvision/svhn",
        num_images=14000,
        samples_per_gpu=128,
        workers_per_gpu=4,
        seed=seed,
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type="TVDatasetSplit",
        base="SVHN",
        split="test",
        data_prefix="data/torchvision/svhn",
        num_images=-1,
        samples_per_gpu=128,
        workers_per_gpu=4,
        seed=seed,
        pipeline=__test_pipeline,
        download=True,
    ),
)
