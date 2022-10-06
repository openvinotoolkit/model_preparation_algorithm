_base_ = [
    './pipelines/coco_ote_pipeline.py'
]

__dataset_type = 'CocoDataset'
__data_root = 'data/coco/'

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_train2017.json',
        img_prefix=__data_root + 'train2017/',
        pipeline=__train_pipeline),
    val=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
        pipeline=__test_pipeline),
    test=dict(
        type=__dataset_type,
        ann_file=__data_root + 'annotations/instances_val2017.json',
        img_prefix=__data_root + 'val2017/',
        test_mode=True,
        pipeline=__test_pipeline)
)
