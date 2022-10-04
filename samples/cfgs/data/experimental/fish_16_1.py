_data_root = '/local/sungmanc/datasets/fish/'
data = dict(
    train=dict(
        ann_file=_data_root + 'annotations/instances_train_16_1.json',
        img_prefix=_data_root + 'images/train/',
        classes=['fish']
    ),
    val=dict(
        ann_file=_data_root + 'annotations/instances_val_100.json',
        img_prefix=_data_root + 'images/val/',
        classes=['fish']
    ),
    test=dict(
        ann_file=_data_root + 'annotations/instances_val_100.json',
        img_prefix=_data_root + 'images/val/',
        classes=['fish']
    )
)