_base_ = [
    '../_base_/default.py',
    '../_base_/logs/tensorboard_logger.py',
    '../_base_/optimizers/sgd.py',
    '../_base_/runners/epoch_runner_cancel.py',
    '../_base_/schedules/plateau.py',
]

optimizer = dict(
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)

lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mAP',
    patience=2,
    iteration_patience=0,
    interval=5,
    min_lr=0.000001,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3)

evaluation = dict(
    interval=5,
    metric=['mIoU', 'mDice'],
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=True,
    iters=0,
    open_layers=[r'\w*[.]?backbone\.aggregator\.', r'\w*[.]?neck\.',
                 r'\w*[.]?decode_head\.', r'\w*[.]?auxiliary_head\.']
)

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=5,
        patience=3,
        iteration_patience=0,
        metric='mAP',
        interval=5,
        priority=75,
    ),
]
