_base_ = [
    '../../../../models/segmentation/_base_/ocr_litehrnet18_mod2.py',
    '../../../stages/segmentation/class_incr.py',
]

task = 'segmentation'

model = dict(
    is_task_adapt=False
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        dataset=dict(
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='RandomResizedCrop', size=(512, 512), interpolation=3),
                dict(type='RandomFlip', prob=0.5),
                dict(type='ProbCompose',
                    transforms=[
                        dict(
                            type='ColorJitter',
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1)
                    ],
                    probs=[0.8]),
                dict(type='RandomGrayscale', p=0.2),
                dict(type='GaussianBlur', kernel_size=23),
                dict(
                    type='Normalize', 
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True
                ),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg']),
            ],
            classes=None
        )
    ),
    val=dict(
        classes=None
    ),
    test=dict(
        classes=None
    )
)

lr_config = dict(warmup_iters=100)

log_config = dict(interval=1)
evaluation = dict(save_best='mDice')

task_adapt = None

seed = 42
deterministic = True
