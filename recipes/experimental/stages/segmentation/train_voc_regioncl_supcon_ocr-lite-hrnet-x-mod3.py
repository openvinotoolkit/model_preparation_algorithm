_base_ = [
    '../_base_/models/segmentors/seg_regioncl_supcon.py',
    '../_base_/models/segmentors/seg_ocr-lite-hrnet-x-mod3.py',
    '../_base_/data/voc_detcon.py',
    '../../../../../../mmsegmentation/submodule/configs/_base_/default_runtime.py',
]

task = 'segmentation'

model = dict(
    head=dict(
        type='RegionCLNonLinearHeadV1',
        in_channels=60,
        hid_channels=60,
        out_channels=128,
        with_avg_pool=True
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
