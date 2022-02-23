_base_ = [
    './finetune.py',
]

model = dict(
    unlabeled_loss_weight=1.0,
)

custom_hooks = [
    dict(
        type='UnbiasedTeacherHook',
        momentum=0.0004,
        start_epoch=2,
        # min_pseudo_label_ratio=0.1,
        min_pseudo_label_ratio=0.0,
    ),
    dict(
        type='LazyEarlyStoppingHook',
        patience=5,
        iteration_patience=1000,
        metric='bbox_mAP',
        interval=1,
        priority=75,
        start=3
    ),
]
