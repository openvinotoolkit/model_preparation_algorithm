_base_ = [
    '../_base_/models/segmentors/seg_detcon_supcon.py',
    '../_base_/models/segmentors/seg_ocr-lite-hrnet-x-mod3.py',
    '../_base_/data/voc_detcon.py',
    '../../../../../../mmsegmentation/submodule/configs/_base_/default_runtime.py',
]

task = 'segmentation'

model = dict(
    downsample=2,
    projector=dict(
        in_channels=60, # after multiple_select, output channel is 60
        hid_channels=120,
        out_channels=60,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=60,
        hid_channels=120,
        out_channels=60,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_avg_pool=False
    ),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
    weight_decay=0.0005
)
optimizer_config = dict(
    grad_clip=dict(
        # method='default',
        max_norm=40,
        norm_type=2
    )
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=False,
    iters=0,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# learning policy
lr_config = dict(
    policy='customcos',
    by_epoch=False,
    periods=[36000],
    min_lr_ratio=1e-3,
    alpha=1.2,
    warmup='cos',
    warmup_iters=4000,
    warmup_ratio=1e-3,
)

# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=40000
)
checkpoint_config = dict(
    by_epoch=False,
    interval=1000
)
evaluation = dict(
    interval=1000,
    metric='mIoU'
)

task_adapt = None

seed = 42
deterministic = True
find_unused_parameters = False
