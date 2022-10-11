_base_ = [
    '../../../recipes/stages/_base_/data/multimodal_dataset.py'
]

_classes = [0, 1, 2, 3, 4]
_tabular_cfg = './petFinder.json'
_modalities = ['tabular']
data = dict(
    train=dict(
        img_data='./data/petFinder/train_images',
        table_data='./data/petFinder/train/train.csv',
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities
    ),
    val=dict(
        img_data='./data/petFinder/val_images',
        table_data='./data/petFinder/val/val.csv',
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities
    ),
    test=dict(
        img_data='./data/petFinder/val_images',
        table_data='./data/petFinder/val/val.csv',
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities
    )
)
