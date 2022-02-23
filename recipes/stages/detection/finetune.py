_base_ = [
    './train.py',
    '../_base_/models/detectors/detector.py'
]

model = dict(super_type='UnbiasedTeacher')  # Used as general framework

custom_hooks = [
    dict(type='DualModelEMAHook',
        momentum=0.0004,
        start_epoch=2,
    ),
    dict(type='EarlyStoppingHook',
        patience=5,
        iteration_patience=1000,
        metric='bbox_mAP',
        interval=1,
        priority=75
    ),
]
