_base_ = [
    './cls_data.py'
]

__train_pipeline = {{_base_.train_rand_pipeline}}

data = dict(
    train=dict(
        type='ClsDirDataset',
        pipeline=__train_pipeline))
