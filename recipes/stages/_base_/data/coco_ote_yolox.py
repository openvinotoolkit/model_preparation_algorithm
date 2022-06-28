_base_ = [
    './data.py',
    './pipelines/coco_ote_yolox_pipeline.py'
]

__dataset_type = 'CocoDataset'
__data_root = 'data/coco/'

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}
__img_scale = {{_base_.img_scale}}

__samples_per_gpu = 2

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=4,
    num_classes=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=__dataset_type,
            ann_file=__data_root + 'annotations/instances_train2017.json',
            img_prefix=__data_root + 'train2017/',
            pipeline=[
                    dict(type='LoadImageFromFile', to_float32=True),
                    dict(type='LoadAnnotations', with_bbox=True)
            ],
        ),
        pipeline=__train_pipeline,
        dynamic_scale=__img_scale,
    ),
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