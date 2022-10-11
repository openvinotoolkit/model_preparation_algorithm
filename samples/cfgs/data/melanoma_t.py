_base_ = [
    '../../../recipes/stages/_base_/data/multimodal_dataset.py',
]

_tabular_cfg = './melanoma.json'
_modalities = ['tabular']
_classes = ['benign', 'malgnant']
_task_type = 'classification'

data = dict(
    train=dict(
        img_data='./data/siim-isic-melanoma-classification/balanced/train',
        train_table_data='./data/siim-isic-melanoma-classification/split_train_balanced.csv',
        val_table_data='./data/siim-isic-melanoma-classification/split_val_balanced.csv',
        is_train=True,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities,
        classes=_classes,
        task_type=_task_type
    ),
    val=dict(
        img_data='./data/siim-isic-melanoma-classification/balanced/val',
        train_table_data='./data/siim-isic-melanoma-classification/split_train_balanced.csv',
        val_table_data='./data/siim-isic-melanoma-classification/split_val_balanced.csv',
        is_train=False,
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities,
        task_type=_task_type
    ),
    test=dict(
        img_data='./data/siim-isic-melanoma-classification/balanced/val',
        train_table_data='./data/siim-isic-melanoma-classification/split_train_balanced.csv',
        val_table_data='./data/siim-isic-melanoma-classification/split_val_balanced.csv',
        is_train=False,
        classes=_classes,
        tabular_cfg=_tabular_cfg,
        modalities=_modalities,
        task_type=_task_type
    )
)
