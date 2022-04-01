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

lr_config = dict(min_lr=1e-06)

evaluation = dict(
    interval=5,
    metric=['mIoU', 'mDice'],
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
    dict(type='LazyEarlyStoppingHook',
         patience=5,
         iteration_patience=1000,
         metric='mIoU',
         interval=1,
         priority=75,
         start=1
         ),
]
