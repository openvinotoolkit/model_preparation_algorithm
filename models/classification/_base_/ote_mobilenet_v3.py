# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OTEMobileNetV3',
        mode='small',
        width_mult=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=576,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
