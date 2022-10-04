_base_ = [
    '../_base_/data/default_ote.py',
    '../_base_/models/detectors/det_default.py',
    '../../../stages/detection/train.py'
]

task = 'detection'

data = dict(
    train=dict(super_type=None),
)

task_adapt = dict(
    type='mpa',
    op='REPLACE',
    efficient_mode=False,
)

runner = dict(
    max_epochs=30
)

evaluation = dict(interval=1, metric='mAP', save_best='mAP')

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=3,
        patience=10,
        iteration_patience=0,
        metric='mAP',
        interval=1,
        priority=75,
    ),
]

lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mAP',
    patience=5,
    iteration_patience=0,
    interval=1,
    min_lr=1e-06,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
)

ignore = True
adaptive_validation_interval = dict(max_interval=5)

load_from=None
