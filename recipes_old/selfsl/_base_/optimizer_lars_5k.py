# optimizer
optimizer = dict(type='LARS',  mode='selfsl', lr=1.0, weight_decay=1e-5, momentum=0.9)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.0001,
    warmup_by_epoch=False)

momentum_config = dict(policy='Pixpro', end_momentum=1.)

# runner settings
runner = dict(type='EpochBasedRunner', max_epochs=5)

