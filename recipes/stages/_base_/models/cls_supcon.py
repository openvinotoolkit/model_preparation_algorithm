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
        type='HybridClsHead',
        num_classes=10,
        in_channels=-1,
        hid_channels=512,
        out_channels=128,
        loss=dict(
            type='SupConLoss',
            temperature=0.07,
            contrast_mode='all',
            base_temperature=0.07,
            lamda=1.0
        )
    )
)
