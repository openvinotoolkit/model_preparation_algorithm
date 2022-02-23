# optimizer settings
_base_ = [
    '../../_base_/optimizer.py'
]

lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.0001
)
