_base_ = [
    './train.py',
    '../_base_/models/segmentors/seg_class_incr.py',
    '../_base_/data/voc_cls_incr.py'
]

optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
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

lr_config = dict(
    metric='mIoU',
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
    type='EpochRunnerWithCancel',
    max_epochs=300
)

checkpoint_config = dict(
    by_epoch=True,
    interval=1
)

evaluation = dict(
    interval=1,
    metric=['mIoU'],
)

seed = 42

task_adapt = dict(
    type='mpa',
    op='MERGE',
)

ignore = True
