_base_ = [
    '../_base_/models/segmentors/seg_detcon_supcon.py',
    '../_base_/data/voc_detcon.py',
    '../../../../../../mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-s-mod2/model.py',
]

task = 'segmentation'

model = dict(
    downsample=8,
    input_transform='multiple_select',
    projector=dict(
        in_channels=60, # after multiple_select, output channel is 60
        hid_channels=120,
        out_channels=60,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=60,
        hid_channels=120,
        out_channels=60,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_avg_pool=False
    ),
)

task_adapt = None

seed = 42
deterministic = True