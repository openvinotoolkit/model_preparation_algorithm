_base_ = [
  '../../../../samples/cfgs/models/backbones/ote_mobilenet_v2_w1.yaml',
  '../../../../recipes/stages/_base_/models/detectors/atss.custom.py'
]

model = dict(
    backbone=dict(out_indices=(2, 3, 4, 5,))
)
