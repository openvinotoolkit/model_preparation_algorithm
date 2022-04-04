_base_ = [
    './train.py',
    '../_base_/models/segmentors/seg_semisl.py'
]

optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(
        # method='adaptive',
        # clip=0.2,
        # method='default', # ?
        max_norm=40,
        norm_type=2
    )
)

lr_config = dict(
    _delete_=True,
    policy='customstep',
    by_epoch=True,
    gamma=0.1,
    step=[200, 250],
    fixed='constant',
    fixed_iters=40,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=80,
    warmup_ratio=1e-2,
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=True,
    iters=40,
    open_layers=[r'\w*[.]?backbone\.aggregator\.', r'\w*[.]?neck\.',
                 r'\w*[.]?decode_head\.', r'\w*[.]?auxiliary_head\.']
)

custom_hooks = [
    dict(
        type='DualModelEMAHook',
        momentum=0.99,
        start_epoch=1,
        src_model_name='model_s',
        dst_model_name='model_t',
    ),
]

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook', commit=False, init_kwargs=dict(name='EXP_NAME'))
    ]
)

dist_params = dict(
    backend='nccl',
    linear_scale_lr=False
)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=300
)

checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5
)

evaluation = dict(
    _delete_=True,
    interval=1,
    metric=['mIoU', 'mDice'],
    rule='greater',
    save_best='mDice'
)

seed = 42
task_adapt = dict(_delete_=True)
