# model settings
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='OmzBackboneCls',
        mode='train',
        model_path='public/efficientnet-b7_auto_aug/FP32/efficientnet-b7_auto_aug.xml',
        last_layer_name='efficientnet-b7/model/blocks_54/Add',
        normalized_img_input=True
    ),
    neck=dict(
        type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=640,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
