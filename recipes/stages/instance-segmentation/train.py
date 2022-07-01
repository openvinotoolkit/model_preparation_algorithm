_base_ = [
    '../_base_/default.py',
    '../_base_/data/data.py',
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
    interval=1,
    metric='mAP',
    classwise=True,
    save_best='mAP'
)

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=3,
        patience=5,
        iteration_patience=1000,
        metric='mAP',
        interval=1,
        priority=75,
    ),
]
