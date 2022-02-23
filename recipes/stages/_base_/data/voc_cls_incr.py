_base_ = [
    './data_seg.py'
]

__dataset_type = 'PascalVOCDataset'
__data_root = 'data/pascal_voc'

data = dict(
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=__dataset_type,
            data_root=__data_root,
            img_dir='train_aug/image',
            ann_dir='train_aug/label',
            split='train_aug.txt',
            # For IL, splited dataset must be set in 'split' like below.
            # split='train_aug/person_seed2_train8.txt',
            classes=['background', 'person', 'car'],
        )
    ),
    val=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir='val/image',
        ann_dir='val/label',
        split='val.txt',
        classes=['background', 'person', 'car'],
    ),
    test=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir='val/image',
        ann_dir='val/label',
        split='val.txt',
        classes=['background', 'person', 'car'],
    )
)
