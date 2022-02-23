_base_ = [
    './pipelines/hflip_resize.py'
]

__dataset_type = 'LwfTaskIncDataset'
__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    pipeline_options=dict(
        Resize=dict(
            size=(256, 128)
        ),
        Normalize=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
    ),
    train=dict(
        type=__dataset_type,
        data_prefix='data/dss18/DSS18_person',
        data_file='data/dss18/csvs/dss18.data.csv',
        ann_file='data/dss18/csvs/dss18.anno.train.csv',
        pipeline=__train_pipeline,
        tasks=dict(
            Age=['Other', 'Senior', 'Kids', 'Unknown'],
        )
    ),
    val=dict(
        type=__dataset_type,
        data_prefix='data/dss18/DSS18_person',
        data_file='data/dss18/csvs/dss18.data.csv',
        ann_file='data/dss18/csvs/dss18.anno.val.csv',
        pipeline=__test_pipeline,
        tasks=dict(
            Age=['Other', 'Senior', 'Kids', 'Unknown'],
            Gender=['Male', 'Female', 'Unknown'],
            Backpack=['Yes', 'No'],
            Longhair=['Yes', 'No', 'Unknown']
        )
    ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=__dataset_type,
        data_prefix='data/dss18/DSS18_person',
        data_file='data/dss18/csvs/dss18.data.csv',
        ann_file='data/dss18/csvs/dss18.anno.test.csv',
        pipeline=__test_pipeline,
        tasks=dict(
            Age=['Other', 'Senior', 'Kids', 'Unknown'],
            Gender=['Male', 'Female', 'Unknown'],
            Backpack=['Yes', 'No'],
            Longhair=['Yes', 'No', 'Unknown']
        )
    )
)
