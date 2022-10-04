_base_ = [
    '../../../../../../samples/cfgs/models/detectors/atss_mv2w1.custom.yaml',
]

model = dict(
    type='DetConBSupCon',
    num_classes=None,
    num_samples=16,
    downsample=4,
    input_transform='resize_concat',
    in_index=[0,1,2,3],
    projector=dict(
        in_channels=sum([24, 32, 96, 320]),
        hid_channels=sum([24, 32, 96, 320]) * 2,
        out_channels=sum([24, 32, 96, 320]),
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    predictor=dict(
        in_channels=sum([24, 32, 96, 320]),
        hid_channels=sum([24, 32, 96, 320]) * 2,
        out_channels=sum([24, 32, 96, 320]),
        norm_cfg=dict(type='BN1d', requires_grad=True),
        with_avg_pool=False
    ),
    detcon_loss_cfg=dict(type='DetConBLoss', temperature=0.1),
    pretrained='https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth',
)