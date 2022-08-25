_base_ = [
    '../_base_/models/segmentors/seg_regioncl_supcon.py',
    '../_base_/data/voc_regioncl.py',
    '../../../../../../mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2/model.py',
]

task = 'segmentation'

model = dict(
    head=dict(
        type='RegionCLNonLinearHeadV1',
        in_channels=40,
        hid_channels=40,
        out_channels=128,
        with_avg_pool=True
    ),
)

task_adapt = None

seed = 42
deterministic = True
