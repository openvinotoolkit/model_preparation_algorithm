# model settings
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='OmzBackboneCls',
        mode='train',
        model_path='public/googlenet-v3/FP32/googlenet-v3.xml',
        last_layer_name='InceptionV3/InceptionV3/Mixed_7c/concat_v2',
        normalized_img_input=True
    ),
    neck=dict(
        type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
