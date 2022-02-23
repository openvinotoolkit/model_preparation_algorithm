# optimizer
optimizer = dict(type='SGD', lr=0.001*2, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    grad_clip=dict(max_norm=40, norm_type=2))
# params_config = dict(
#     type='FreezeLayers',
#     by_epoch=True,
#     iters=0,
#     open_layers=[
#         'backbone\\.aggregator\\.', 'neck\\.', 'decode_head\\.',
#         'auxiliary_head\\.'
#     ])

custom_hooks = [
    dict(
        type='SwitchHook',
        switch_point=9999999,  # set 1 if you want CPS
        momentum=0.01,
        start_epoch=1,
        src_model_name='model_s',
        dst_model_name='model_t'
         ),
]

lr_config = dict(
    policy='customstep',
    gamma=0.1,
    by_epoch=True,
    step=[200, 250],
    # fixed='constant',
    # fixed_iters=0,
    # fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=80,
    warmup_ratio=0.01)
