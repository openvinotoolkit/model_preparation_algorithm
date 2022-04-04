# model settings
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='OmzBackboneCls',
        mode='train',
        model_path='public/vgg19-caffe2/FP32/vgg19-caffe2.xml',
        last_layer_name='conv5_4_2',
        normalized_img_input=True
    ),
    neck=dict(
        type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
