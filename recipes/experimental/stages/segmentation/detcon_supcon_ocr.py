_base_ = [
    '../../../stages/segmentation/train.py',
    '../_base_/models/segmentors/seg_detcon_supcon_ocr.py',
    '../_base_/data/voc_detcon.py'
]

optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=1e-3,
    eps=1e-08,
    weight_decay=0.0
)

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(
        # method='adaptive',
        # clip=0.2,
        # method='default',
        max_norm=40,
        norm_type=2
    )
)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
    ]
)

dist_params = dict(
    backend='nccl',
    linear_scale_lr=False
)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=300
)

checkpoint_config = dict(
    by_epoch=True,
    interval=1
)

evaluation = dict(
    interval=1,
    metric=['mDice', 'mIoU'],
    show_log=True
)

task_adapt = None

seed = 42
deterministic = True
find_unused_parameters = False

ignore = True
