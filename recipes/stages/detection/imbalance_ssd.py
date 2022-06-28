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

evaluation = dict(interval=1, metric='mAP', save_best='mAP')

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

ignore = False
