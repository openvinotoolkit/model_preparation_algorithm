_base_ = [
    './efficientnetb2b_maskrcnn.py'
]

model = dict(
    type='CustomMaskRCNN',
)
