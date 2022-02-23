_base_ = [
    './pipelines/cls_incr_cityscapes.py'
]

__dataset_type = 'CityscapesDataset'
__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

seed = 1234

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=__dataset_type,
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=__train_pipeline,
        classes=['car', 'building']
    ),
    val=dict(
        type=__dataset_type,
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=__test_pipeline,
        classes=['car', 'building']
    ),
    test=dict(
        type=__dataset_type,
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=__test_pipeline,
        classes=['car', 'building']
    )
)
