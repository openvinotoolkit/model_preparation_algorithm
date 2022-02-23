# model settings
_base_ = [
    './model_lite_hrnet_ote_ce.py'
]
# pre-trained params settings
ignore_keys = [r'^backbone\.increase_modules\.', r'^backbone\.increase_modules\.',
               r'^backbone\.downsample_modules\.', r'^backbone\.final_layer\.',
               r'^head\.']

__norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    backbone=dict(norm_cfg=__norm_cfg),
    decode_head=[
        dict(type='FCNHeadMPA',
             in_channels=40,
             in_index=0,
             channels=40,
             input_transform=None,
             kernel_size=1,
             num_convs=0,
             concat_input=False,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=__norm_cfg,
             align_corners=False,
             enable_out_norm=False,
             loss_decode=[
                dict(type='CrossEntropyLossMPA',
                     use_sigmoid=False,
                     loss_jitter_prob=0.01,
                     sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                     loss_weight=1.0)
                         ]),
        dict(type='OCRHeadMPA',
             in_channels=40,
             in_index=0,
             channels=40,
             ocr_channels=40,
             sep_conv=True,
             input_transform=None,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=__norm_cfg,
             align_corners=False,
             enable_out_norm=True,
             loss_decode=[
                 dict(type='AMSoftmaxLoss',
                      scale_cfg=dict(
                          type='PolyScalarScheduler',
                          start_scale=30,
                          end_scale=5,
                          num_iters=400,
                          power=1.2
                      ),
                      margin_type='cos',
                      margin=0.5,
                      gamma=0.0,
                      t=1.0,
                      target_loss='ce',
                      pr_product=False,
                      conf_penalty_weight=dict(
                          type='PolyScalarScheduler',
                          start_scale=0.2,
                          end_scale=0.15,
                          num_iters=500,
                          power=1.2
                      ),
                      loss_jitter_prob=0.01,
                      border_reweighting=False,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]
             )
        ],
    # model training and testing settings
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode='whole')
)
