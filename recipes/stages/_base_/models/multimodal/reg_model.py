_base_ = '../model.py'

model = dict(
    type='MultimodalModel',
    task='VisionTabularRegression',
    pretrained=None,
    visual_encoder=dict(),
    textual_encoder=dict(),
    tabular_encoder=dict(),
    head=dict(),
)

checkpoint_config = dict(
    type='CheckpointHookWithValResults'
)