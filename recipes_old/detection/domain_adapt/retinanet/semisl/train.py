_base_ = '../finetune/train.py'

name = 'det-da-retina-semisl-train'

model = dict(
    unlabeled_loss_weight=1.0,
    bbox_head=dict(
        type='CustomRetinaHead',
    ),
)

data = dict(
    train=dict(
        use_unlabeled=True,
    ),
)

custom_hooks = [
    #dict(type='NumClassCheckHook'),
    dict(type='UnbiasedTeacherHook',
        momentum=0.0004,
        start_epoch=2,
        #min_pseudo_label_ratio=0.1,
        min_pseudo_label_ratio=0.0,
    ),
    dict(type='EarlyStoppingHook',
        patience=5,
        iteration_patience=1000,
        metric='bbox_mAP',
        interval=1,
        priority=75
    ),
]

