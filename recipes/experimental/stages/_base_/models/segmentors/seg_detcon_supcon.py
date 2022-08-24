norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 21
model = dict(
    type='DetConSupCon',
    num_stages=2,
    pretrained=None,
    num_classes=num_classes,
    num_samples=16,
    downsample=4,
    input_transform='resize_concat',
    in_index=[0,1,2,3],
    backbone=dict(
        norm_cfg=norm_cfg,
    ),
    projector=dict(
        in_channels=sum([40, 40, 80, 160]), # 320
        hid_channels=sum([40, 40, 80, 160]) * 2, # 640
        out_channels=num_classes,
        norm_cfg=norm_cfg,
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=num_classes, # 320
        hid_channels=sum([40, 40, 80, 160]) * 2, # 640
        out_channels=num_classes,
        norm_cfg=norm_cfg,
        with_avg_pool=False
    ),
    train_cfg=dict(
        mix_loss=dict(enable=False, weight=0.1)
    ),
    detcon_loss_cfg=dict(type='DetConBLoss', temperature=0.1, use_replicator_loss=True)
)
evaluation = dict(
    metric='mIoU'
)

find_unused_parameters = True
