# model settings
model = dict(
    type='Pixpro',
    pretrained='torchvision://resnet50',
    base_momentum=0.99,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='Projector',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256
    ),
    head=dict(
        type='PPM',
        sharpness=2
    ),
    pos_ratio=0.7
)
