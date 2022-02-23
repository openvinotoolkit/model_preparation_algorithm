_base_ = [
    '../../_base_/base.py'
]

# yapf:disable
log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
log_level = 'INFO'
# yapf:enable
