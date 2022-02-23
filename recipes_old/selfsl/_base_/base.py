# checkpoint saving
#checkpoint_config = dict(interval=5000)
#checkpoint_config = dict(interval=462)
checkpoint_config = dict(interval=5)

# log setting
log_config = dict(
    interval=100,
    ignore_last=False,
    hooks=[
        dict(type='TextLoggerHook', ignore_last=False),
        dict(type='TensorboardLoggerHook')
    ])
log_level = 'INFO'

# runtime settings
dist_params = dict(backend='nccl', linear_scale_lr=True)
cudnn_benchmark = True

workflow = [('train', 1)]
seed = 5
