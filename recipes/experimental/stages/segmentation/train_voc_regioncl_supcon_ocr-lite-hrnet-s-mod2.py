_base_ = [
    '../_base_/models/segmentors/seg_regioncl_supcon.py',
    '../_base_/data/voc_regioncl.py',
    '../../../../../../mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-s-mod2/model.py',
]

task = 'segmentation'

model = dict(
    pretrained='https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetsv2_imagenet1k_rsc.pth',
    head=dict(
        type='RegionCLNonLinearHeadV1',
        in_channels=60,
        hid_channels=60,
        out_channels=128,
        with_avg_pool=True
    ),
)

task_adapt = None

seed = 42
deterministic = True
