_base_ = '../../../../models/detection/ote_custom_od_atss_mv2_21aug.py'

name = 'det-da-atss-train'
type = 'DetectionTrainer'

task_adapt = dict(
    op='REPLACE',
)

hparams = dict(
    dummy=0,
)

data = dict(
    samples_per_gpu=2,  # 9 in OTE setting
    train=dict(
        _delete_=True,
        type='PseudoSemiDataset',
        labeled_percent=100.0,
        use_unlabeled=False,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(
                type='Resize',
                img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672),
                           (992, 800)],
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
    ),
    val=dict(samples_per_gpu=2),
    test=dict(samples_per_gpu=2),
)

optimizer = dict(lr=0.001)  # 0.009 in OTE setting
lr_config = dict(min_lr=1e-06)  # 9e-06 in OTE setting

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
)

gpu_ids = range(0, 1)
load_from = None
work_dir = None
