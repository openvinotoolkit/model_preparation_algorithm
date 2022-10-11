_base_ = [
    '../../../recipes/stages/_base_/data/multimodal_dataset.py',
    '../../../recipes/stages/_base_/data/data.py',
    '../../../recipes/stages/_base_/data/pipelines/semisl_pipeline.py',
]

__img_train_pipeline = {{_base_.train_pipeline}}
__img_test_pipeline = {{_base_.test_pipeline}}

_classes = [0, 1, 2, 3, 4]
_tabular_cfg = './petFinder.json'
_modalities = ['vision']
data = dict(
    train=dict(
        img_data='./data/petFinder/train_images',
        img_pipeline=__img_train_pipeline,
        table_data='./data/petFinder/train/train.csv',
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities
    ),
    val=dict(
        img_data='./data/petFinder/val_images',
        img_pipeline=__img_test_pipeline,
        table_data='./data/petFinder/val/val.csv',
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities
    ),
    test=dict(
        img_data='./data/petFinder/val_images',
        img_pipeline=__img_test_pipeline,
        table_data='./data/petFinder/val/val.csv',
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities
    )
)
