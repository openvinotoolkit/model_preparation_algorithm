# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)
optimizer_config = dict(
    grad_clip=dict(
        # method='adaptive',
        # clip=0.2,
        # method='default', # ?
        max_norm=40,
        norm_type=2
    )
)

# learning policy
lr_config = dict(
    policy='customstep',
    by_epoch=True,
    gamma=0.1,
    step=[200, 250],
    fixed='constant',
    fixed_iters=40,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=80,
    warmup_ratio=1e-2,
)
