_base_ = './schedule.py'

lr_config = dict(
    policy='ReduceLROnPlateau',
    patience=3,
    interval=1,
    iteration_patience=200,
    metric='accuracy',
    min_lr=1e-05,
    warmup='linear'
)
