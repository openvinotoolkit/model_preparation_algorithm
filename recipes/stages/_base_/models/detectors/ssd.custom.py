_base_ = [
    './ssd.ote.py'
]

model = dict(
    type='CustomSingleStageDetector',
    bbox_head=dict(type='CustomSSDHead',),
)

__img_norm_cfg = dict(
    mean=[0, 0, 0],
    std=[255, 255, 255],
    to_rgb=True,
)

data = dict(
    pipeline_options=dict(
        MinIouRandomCrop=dict(min_crop_size=0.1),
        Resize=dict(img_scale=(864, 864)),
        MultiScaleFlipAug=dict(
            img_scale=(864, 864),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Normalize', **__img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ],
        ),
    ),
)

ignore = False
