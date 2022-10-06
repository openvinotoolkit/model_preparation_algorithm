_base_ = [
    '../_base_/data/coco_ote.py',
    '../../../stages/detection/train.py',
    '../../../../samples/cfgs/models/detectors/atss_mv2w1.custom.yaml',
]

task = 'detection'

task_adapt = dict(
    type='mpa',
    op='REPLACE',
    efficient_mode=False,
)

runner = dict(max_epochs=200)

optimizer = dict(lr=0.004)

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
    metric='mAP',
    patience=5,
    warmup_iters=3)

model = dict(
    backbone=dict(norm_eval=False)
)

cudnn_benchmark = True

seed = 42
deterministic = True

hparams = dict(dummy=0)

ignore = True
adaptive_validation_interval = dict(max_interval=5)
fp16 = dict(loss_scale=512.0)

load_from = None
