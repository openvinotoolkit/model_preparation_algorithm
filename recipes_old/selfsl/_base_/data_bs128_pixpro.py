# dataset settings
data = dict(
    samples_per_gpu=128,
    #samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='PixproDataset',
        datasource=dict(
            cfg=dict(type=''),
            reg=None
        ),
        crop=0.08,
        img_size=224,
        aug_type='BYOL'
    )
)
