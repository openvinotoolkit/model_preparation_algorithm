_base_ = [
    './data_seg.py'
]

__dataset_type = 'CityscapesDataset'
__data_root = 'data/cityscapes'

data = dict(
    train=dict(
        dataset=dict(
            type=__dataset_type,
            data_root=__data_root,
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            # For IL, splited dataset must be set in 'split' like below.
            # split='gtFine_car-building/1-32/seed0_train8.txt',
            classes=['background', 'car', 'building'],
        )
    ),
    val=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        classes=['background', 'car', 'building'],
    ),
    test=dict(
        type=__dataset_type,
        data_root=__data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        classes=['background', 'car', 'building'],
    )
)
