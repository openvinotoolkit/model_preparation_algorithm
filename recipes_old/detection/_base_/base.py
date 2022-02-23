_base_ = [
    '../../_base_/base.py'
]

# yapf:disable
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
# yapf:enable
