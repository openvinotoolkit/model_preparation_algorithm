norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 21
model = dict(
    type='RegionCLMSupCon',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_cutmix=1.0,
    cutmix_alpha=1.0,
    cutMixUpper=14,
    cutMixLower=6,
    num_classes=num_classes,
    input_transform='resize_concat',
    in_index=[0,1,2,3],
    head=dict(
        type='RegionCLNonLinearHeadV1',
        in_channels=sum([40, 40, 80, 160]), # 320
        hid_channels=sum([40, 40, 80, 160]), # 320
        out_channels=128,
        with_avg_pool=True
    ),
    regioncl_loss_cfg=dict(type='ContrastiveLoss', temperature=0.2),
    train_cfg=dict(
        mix_loss=dict(enable=False, weight=0.1)
    ),
)
