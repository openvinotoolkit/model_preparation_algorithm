# model settings
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='OmzBackboneCls',
        mode='train',
        model_path='public/googlenet-v2/FP32/googlenet-v2.xml',
        last_layer_name='inception_5b/output',
        normalized_img_input=True
    ),
    neck=dict(
        type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
