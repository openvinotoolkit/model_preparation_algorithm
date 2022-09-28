# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OTEEfficientNetV2',
        pretrained=True,
        version='s_21k'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='CustomLinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
