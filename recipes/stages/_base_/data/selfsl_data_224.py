"""Base Self-SL dataset."""

__resize_target_size = 224
__img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

__train_pipeline_v0 = [
    dict(type='RandomResizedCrop', size=__resize_target_size),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')],
         p=0.),
    dict(type='Normalize',  **__img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
__train_pipeline_v1 = [
    dict(type='RandomResizedCrop', size=__resize_target_size),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.1),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')],
         p=0.2),
    dict(type='Normalize',  **__img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type='SelfSLDataset',
        pipeline=dict(
            view0=__train_pipeline_v0,
            view1=__train_pipeline_v1,
        )
    )
)
