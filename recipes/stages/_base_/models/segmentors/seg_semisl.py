_base_ = './encoder_decoder.ote.py'

__norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CutmixSegNaive',
    ori_type='SemiSLSegmentor',
    unsup_weight=0.1,
    num_stages=2,
    decode_head=[
        dict(type='FCNHead',
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
                dict(type='CrossEntropyLoss',
                     loss_jitter_prob=0.01,
                     sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                     loss_weight=1.0)
             ]),
        dict(type='OCRHead',
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
                          by_epoch=True,
                          num_iters=250,
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
                          by_epoch=True,
                          num_iters=200,
                          power=1.2
                      ),
                      loss_jitter_prob=0.01,
                      border_reweighting=False,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0)
             ])
    ],
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode='whole', output_scale=5.0),
)

find_unused_parameters = True
