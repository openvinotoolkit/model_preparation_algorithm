_base_ = [
    '../_base_/models/segmentors/seg_regioncl_supcon.py',
    '../_base_/data/voc_regioncl.py',
    '../../../../models/segmentation/_base_/ocr_litehrnet_x_mod3.py',
    '../../../stages/segmentation/class_incr.py',
]

task = 'segmentation'

model = dict(
    pretrained='https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetxv3_imagenet1k_rsc.pth',
    is_task_adapt=False,
    input_transform=None,
    in_index=None,
    head=dict(
        type='RegionCLNonLinearHeadV1',
        in_channels=60,
        hid_channels=256,
        out_channels=128,
        with_avg_pool=True
    ),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        dataset=dict(
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
