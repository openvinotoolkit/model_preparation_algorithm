_base_ = [
    './classifiers/classifier.py'
]

model = dict(
    type='SupConClassifier',
    backbone=dict(
        type='OTEMobileNetV3',
        pretrained=True,
        mode='small',
        width_mult=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SupConClsHead',
        num_classes=10,
        in_channels=-1,
        hid_channels=512,
        out_channels=128,
        loss=dict(
            type='BarlowTwinsLoss',
            off_diag_penality=1. / 128.
        )
    )
)
