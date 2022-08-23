_base_ = '../../../../../stages/_base_/models/segmentors/encoder_decoder.ote.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 21
model = dict(
    type='DetConSupCon',
    num_stages=2,
    pretrained=None,
    num_classes=num_classes,
    num_samples=16,
    downsample=4,
    input_transform='resize_concat',
    in_index=[0,1,2,3],
    use_decode_head=True,
    backbone=dict(
        norm_cfg=norm_cfg,
    ),
    decode_head=[
        dict(type='FCNHead',
             in_channels=[40,40,80,160],
             in_index=[0,1,2,3],
             channels=sum([40,40,80,160]),
             input_transform='resize_concat',
             kernel_size=1,
             num_convs=0,
             concat_input=False,
             dropout_ratio=-1,
             num_classes=num_classes,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_out_norm=False,
             loss_decode=[
                dict(type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_jitter_prob=0.01,
                    sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                    loss_weight=1.0),
             ]),
        dict(type='OCRHead',
             in_channels=[40,40,80,160],
             in_index=[0,1,2,3],
             channels=sum([40,40,80,160]),
             ocr_channels=sum([40,40,80,160]),
             sep_conv=True,
             input_transform='resize_concat',
             dropout_ratio=-1,
             num_classes=num_classes,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_out_norm=True,
             loss_decode=[
                 dict(type='AMSoftmaxLoss',
                      scale_cfg=dict(
                          type='PolyScalarScheduler',
                          start_scale=30,
                          end_scale=5,
                          num_iters=30000,
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
                          num_iters=20000,
                          power=1.2
                      ),
                      loss_jitter_prob=0.01,
                      border_reweighting=False,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
    ],
    projector=dict(
        in_channels=sum([40, 40, 80, 160]), # 320
        hid_channels=sum([40, 40, 80, 160]) * 2, # 640
        out_channels=num_classes,
        norm_cfg=norm_cfg,
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=num_classes, # 320
        hid_channels=sum([40, 40, 80, 160]) * 2, # 640
        out_channels=num_classes,
        norm_cfg=norm_cfg,
        with_avg_pool=False
    ),
    train_cfg=dict(
        mix_loss=dict(enable=False, weight=0.1)
    ),
    detcon_loss_cfg=dict(type='DetConBLoss', temperature=0.1, use_replicator_loss=True)
)
evaluation = dict(
    metric='mIoU'
)

find_unused_parameters = True
