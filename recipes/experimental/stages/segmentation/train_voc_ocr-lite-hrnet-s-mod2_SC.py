_base_ = [
    '../../../../models/segmentation/_base_/ocr_litehrnet_s_mod2.py',
    '../../../stages/segmentation/class_incr.py',
]

task = 'segmentation'

model = dict(
    is_task_adapt=False
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        dataset=dict(
            classes=None
        )
    ),
    val=dict(
        classes=None
    ),
    test=dict(
        classes=None
    )
)

lr_config = dict(warmup_iters=100)

log_config = dict(interval=1)
evaluation = dict(save_best='mDice')

task_adapt = None

seed = 42
deterministic = True
