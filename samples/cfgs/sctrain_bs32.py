# dataset settings
img_per_gpu = 32
dataset_type = 'CSVDatasetCls'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Resize', size=(255, 255)),
    dict(type='RandomCrop', size=(239, 239)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(255, 255)),
    dict(type='CenterCrop', crop_size=(239, 239)),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
classes = ['Train Absent', 'Train Arriving', 'Door Closed', 'Door Open']
data = dict(
    samples_per_gpu=img_per_gpu,
    workers_per_gpu=2,
    num_classes=4,
    train=dict(
        type=dataset_type,
        data_prefix='./data_links/sc/sc-train4/',
        data_file='./data_links/sc/sc-train4/train4.train.csv',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='./data_links/sc/sc-train4/',
        data_file='./data_links/sc/sc-train4/train4.val.csv',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='./data_links/sc/sc-train4/',
        data_file='./data_links/sc/sc-train4/train4.test.csv',
        classes=classes,
        pipeline=test_pipeline))
