# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/pascal_voc'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

img_scale = (544, 544)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='SelfSLCompose',
         pipeline1=[
             dict(type='RandomResizedCrop', size=crop_size, interpolation=3),
             dict(type='RandomAppliedTrans',
                  transforms=[
                      dict(
                          type='ColorJitter',
                          brightness=0.4,
                          contrast=0.4,
                          saturation=0.4,
                          hue=0.1)
                  ],
                  p=0.8),
             dict(type='RandomGrayscale', p=0.2),
             dict(type='RandomAppliedTrans',
                  transforms=[
                    dict(
                        type='GaussianBlur', 
                        kernel_size=23)
                  ],
                  p=0.5),
             dict(type='RandomFlip', prob=0.5),
             dict(type='Normalize', **img_norm_cfg)
         ],
         pipeline2=[
             dict(type='RandomResizedCrop', size=crop_size, interpolation=3),
             dict(type='RandomAppliedTrans',
                  transforms=[
                      dict(
                          type='ColorJitter',
                          brightness=0.4,
                          contrast=0.4,
                          saturation=0.4,
                          hue=0.1)
                  ],
                  p=0.8),
             dict(type='RandomGrayscale', p=0.2),
             dict(type='RandomAppliedTrans',
                  transforms=[
                    dict(
                        type='GaussianBlur', 
                        kernel_size=23)
                  ],
                  p=0.5),
             dict(type='RandomFlip', prob=0.5),
             dict(type='Normalize', **img_norm_cfg)
         ]
    ),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        adaptive_repeat=True,
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train_aug/image',
            ann_dir='train_aug/label',
            split='train_aug.txt',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/label',
        split='val.txt',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/label',
        split='val.txt',
        pipeline=test_pipeline
    )
)
