_base_ = [
    '../_base_/models/segmentors/seg_detcon_supcon.py',
    '../_base_/data/voc_detcon.py',
    '../../../../models/segmentation/_base_/ocr_litehrnet18_mod2.py',
    '../../../stages/segmentation/class_incr.py',
]

task = 'segmentation'

model = dict(
    is_task_adapt=False,
    input_transform='multiple_select',
    projector=dict(
        in_channels=40, # after multiple_select, output channel is 40
        hid_channels=256,
        out_channels=128,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=128,
        hid_channels=256,
        out_channels=128,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
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
