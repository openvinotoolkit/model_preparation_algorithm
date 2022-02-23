_base_ = './cifar10split_224_bs16.py'

data = dict(
    pipeline_options=dict(
        RandomResizedCrop=dict(
            size=(112, 112)
        )
    ),
    unlabeled=dict(
        type='TVDatasetSplit',
        base='CIFAR10',
        train=True,
        data_prefix='data/torchvision/cifar10',
        num_images=1000,
        download=True
    )
)
