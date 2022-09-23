_base_ = [
    './classifiers/classifier.py'
]

model = dict(
    type='SupConClassifier',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SupConClsHead',
        num_classes=10,
        in_channels=-1,
        hid_channels=1024,
        out_channels=128,
        loss=dict(
            type='SupConLoss',
            temperature=0.07,
            contrast_mode='all',
            base_temperature=0.07
        )
    )
)
