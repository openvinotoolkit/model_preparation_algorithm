_base_ = [
    './resnet50_maskrcnn.py'
]

model = dict(
    type='CustomMaskRCNNTileOptimised',
    roi_head=dict(
        type='CustomRoIHead',
    )
)
