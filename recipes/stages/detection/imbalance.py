_base_ = [
    './train.py',
    '../_base_/data/coco_ote.py',
    '../_base_/models/detectors/detector.py'
]

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

evaluation = dict(interval=5, metric='mAP', save_best='mAP')

custom_hooks = [
    dict(
        type='LazyEarlyStoppingHook',
        start=5,
        patience=2,
        iteration_patience=0,
        metric='mAP',
        interval=5,
        priority=75,
    ),
]

lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='bbox_mAP',
    patience=5,
    iteration_patience=0,
    interval=6,
    min_lr=1e-06,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
)

ignore = True
