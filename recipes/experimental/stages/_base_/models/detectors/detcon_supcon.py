model = dict(
    type='DetConBSupCon',
    num_classes=None,
    num_samples=16,
    downsample=4,
    input_transform='resize_concat',
    in_index=[0,1,2,3],
    projector=dict(
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=256,
        hid_channels=4096,
        out_channels=256,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    detcon_loss_cfg=dict(type='DetConBLoss', temperature=0.1),
)