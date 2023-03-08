_base_ = [
    './efficientnetb2b_maskrcnn.py'
]

model = dict(
    type='CustomMaskRCNNTileOptimised',
    roi_head=dict(
        type='CustomRoIHead',
    )
)
