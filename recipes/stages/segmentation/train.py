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

# learning policy
lr_config = dict(
    policy='customstep',
    gamma=0.1,
    by_epoch=True,
    step=[400, 500],
    fixed='constant',
    fixed_iters=0,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=80,
    warmup_ratio=1e-2,
)

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
         patience=10,
         iteration_patience=0,
         metric='mIoU',
         interval=1,
         priority=75,
         start=1
         ),
]
