_base_ = './schedule.py'

lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='bbox_mAP',
    patience=3,
    iteration_patience=30,
    interval=1,
    min_lr=1e-06,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
)
