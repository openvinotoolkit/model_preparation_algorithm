_base_ = './single_stage_detector.py'

__width_mult = 1.0

model = dict(
    bbox_head=dict(
        type='SSDHead',
        num_classes=80,
        in_channels=(int(__width_mult * 96), int(__width_mult * 320)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=[
                [
                    52.035592912165924, 119.0011755748901, 85.9695219479953, 161.82547427262662
                ],
                [
                    132.13386951447404, 169.9169262595566, 239.09752064617635, 170.48677498949777, 282.8843922220859
                ],
            ],
            heights=[
                [
                    45.028670725152395, 83.00777237903321, 118.4496923761446, 87.73250317964613
                ],
                [
                    123.8605041031881, 135.35997027152533, 151.62548586157962, 218.2579087349336, 227.57156506277943
                ],
            ],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2), ),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=False,
    ),
    train_cfg=dict(
        assigner=dict(
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
        ),
        use_giou=False,
        use_focal=False,
    )
)
