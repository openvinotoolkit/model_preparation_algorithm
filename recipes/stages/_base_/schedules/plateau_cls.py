_base_ = './schedule.py'

lr_config = dict(
    policy='ReduceLROnPlateau',
    patience=5,
    interval=1,
    iteration_patience=150,
    metric='accuracy_top-1',
    min_lr=1e-05,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_ratio=0.1,
)
