# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='Fixed')
momentum_config = dict(policy='BYOL', end_momentum=1.)

# runner settings
runner = dict(type='EpochBasedRunner', max_epochs=120)
