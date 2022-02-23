_base_ = '../finetune/train.py'

name = 'det-da-atss-semisl-train'

model = dict(
    unlabeled_loss_weight=1.0,
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
]

