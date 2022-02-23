# model settings
model = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.996,
    backbone=dict(
        cfg=dict(type=''),
        reg=None
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=True
    ),
    head=dict(
        type='LatentPredictHead',
        size_average=True,
        predictor=dict(
            type='NonLinearNeck',
            in_channels=128,
            hid_channels=512,
            out_channels=128,
            with_avg_pool=False
        )
    )
)
