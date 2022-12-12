_base_ = [
    './pipelines/seg_semisl.py'
]

__dataset_type = ''
__data_root = ''
__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}
__unlabeled_pipeline = {{_base_.unlabeled_pipeline}}

data = dict(
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=__dataset_type,
            data_root=__data_root,
            pipeline=__train_pipeline
        )
    ),
    val=dict(
        type=__dataset_type,
        data_root=__data_root,
        pipeline=__test_pipeline
    ),
    test=dict(
        type=__dataset_type,
        data_root=__data_root,
        pipeline=__test_pipeline
    ),
    unlabeled=dict(
        type=__dataset_type,
        data_root=__data_root,
        pipeline=__unlabeled_pipeline,
        # cutmix=True
    )
)
