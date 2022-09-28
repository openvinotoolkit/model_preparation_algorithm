_base_ = [
    '../_base_/models/segmentors/seg_detcon_supcon.py',
    '../_base_/data/voc_detcon.py',
    '../../../../models/segmentation/_base_/ocr_litehrnet_s_mod2.py',
    '../../../stages/segmentation/class_incr.py',
]

task = 'segmentation'

model = dict(
    pretrained='https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetsv2_imagenet1k_rsc.pth',
    is_task_adapt=False,
    downsample=8,
    input_transform='resize_concat',
    in_index=[0,1,2],
    projector=dict(
        in_channels=420,
        hid_channels=256,
        out_channels=128,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=128,
        hid_channels=256,
        out_channels=128,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    loss_weights={
        'decode.loss_seg': 1.,
        'detcon': 1.,
    }
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

custom_hooks = [
    dict(type='LazyEarlyStoppingHook',
         patience=8,
         iteration_patience=0,
         metric='mDice',
         interval=1,
         priority=75,
         start=1
         ),
    dict(type='SwitchPipelineHook',
         interval=1
         ),
]

task_adapt = None

seed = 42
deterministic = True
