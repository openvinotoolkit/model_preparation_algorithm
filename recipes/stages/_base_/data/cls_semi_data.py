_base_ = [
    './data.py',
    './pipelines/semisl_pipeline.py'
]

__dataset_type = 'ClsDirDataset'

__train_pipeline_strong = {{_base_.train_pipeline_strong}}
__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

__samples_per_gpu = 32

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        pipeline=__train_pipeline),
    unlabeled=dict(
        type=__dataset_type,
        pipeline=dict(
                weak=__train_pipeline,
                strong=__train_pipeline_strong
        )),
    val=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=__test_pipeline),
    test=dict(
        type=__dataset_type,
        test_mode=True,
        pipeline=__test_pipeline)
)
