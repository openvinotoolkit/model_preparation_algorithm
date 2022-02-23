_base_ = '../../../../models/detection/ote_custom_od_vfnet_r50_21aug.py'

name = 'det-da-vfnet-train'
type = 'DetectionTrainer'

task_adapt = dict(
    op='REPLACE',
)

hparams = dict(
    dummy=0,
)

data = dict(
    samples_per_gpu=2,  # 4 in OTE setting
    train=dict(
        type='PseudoSemiDataset',
        labeled_percent=100.0,
        use_unlabeled=False,
    ),
    val=dict(samples_per_gpu=2),
    test=dict(samples_per_gpu=2),
)

optimizer = dict(lr=0.001)  # 0.01 in OTE setting
lr_config = dict(min_lr=1e-06)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
)

gpu_ids = range(0, 1)  # range(0, 2) in OTE setting
load_from = None
work_dir = None
