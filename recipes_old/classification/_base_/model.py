# model settings
_base_ = [
    '../../_base_/model.py'
]

model = dict(
    type='ImageClassifier',
    task='classification',
    backbone=dict(),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        in_channels=1280,
        num_classes=-1,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)
    )
)
