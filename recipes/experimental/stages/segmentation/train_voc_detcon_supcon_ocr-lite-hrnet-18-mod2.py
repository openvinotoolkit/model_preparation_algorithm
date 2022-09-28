_base_ = [
    '../_base_/models/segmentors/seg_detcon_supcon.py',
    '../_base_/data/voc_detcon.py',
    '../../../../../../mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2/model.py',
]

task = 'segmentation'

model = dict(
    pretrained='https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth',
    input_transform='multiple_select',
    projector=dict(
        in_channels=40, # after multiple_select, output channel is 40
        hid_channels=80,
        out_channels=40,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=40,
        hid_channels=80,
        out_channels=40,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        with_avg_pool=False
    ),
)

task_adapt = None

seed = 42
deterministic = True
