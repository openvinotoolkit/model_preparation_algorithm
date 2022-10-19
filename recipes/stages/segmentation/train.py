_base_ = [
    '../_base_/default.py',
    '../_base_/logs/tensorboard_logger.py',
    '../_base_/optimizers/sgd.py',
    '../_base_/runners/seg_runner.py',
    '../_base_/schedules/plateau.py',
]

optimizer = dict(
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)

lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mDice',
    patience=5,
    iteration_patience=0,
    interval=1,
    min_lr=0.000001,
    warmup='linear',
    warmup_iters=80,
    warmup_ratio=1.0 / 3)

evaluation = dict(
    interval=1,
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
    dict(type='LazyEarlyStoppingHook',
         patience=8,
         iteration_patience=0,
         metric='mDice',
         interval=1,
         priority=75,
         start=1
         ),
]
