# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', by_epoch=True)
    ])
# yapf:enable
dist_params = dict(backend='nccl', linear_scale_lr=False)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=5)
evaluation = dict(by_epoch=True, interval=1, rule='greater', metric=['mIoU', 'mDice'], save_best='mDice')
seed = 42
