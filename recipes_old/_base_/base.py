# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
log_level = 'INFO'

dist_params = dict(backend='nccl', linear_scale_lr=True)
cudnn_benchmark = True

checkpoint_config = dict(interval=1)

runner = dict(
    type='EpochBasedRunner',
    max_epochs=100
)

workflow = [('train', 1)]

seed = 5
