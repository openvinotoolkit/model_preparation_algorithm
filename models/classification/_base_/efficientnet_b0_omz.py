# model settings
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='OmzBackboneCls',
        mode='train',
        model_path='public/efficientnet-b0/FP32/efficientnet-b0.xml',
        last_layer_name='efficientnet-b0/model/blocks_15/tpu_batch_normalization_2/batchnorm/add_1',
        normalized_img_input=True
    ),
    neck=dict(
        type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=320,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
