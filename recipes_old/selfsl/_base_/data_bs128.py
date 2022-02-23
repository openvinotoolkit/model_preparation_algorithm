# dataset settings
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', interpolation=3, size=(112, 112)),  # bicubic
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
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type='BYOLDataset',
        datasource=dict(
            cfg=dict(type=''),
            reg=None
        ),
        # pipeline1=train_pipeline,
        # pipeline2=train_pipeline
        pipeline=dict(
            view0=train_pipeline,
            view1=train_pipeline
        )
    )
)
