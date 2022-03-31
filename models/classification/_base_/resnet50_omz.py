# model settings
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='OmzBackboneCls',
        mode='train',
        model_path='public/resnet-50-caffe2/FP32/resnet-50-caffe2.xml',
        last_layer_name='gpu_0/res5_2_branch2c_bn_3',
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
