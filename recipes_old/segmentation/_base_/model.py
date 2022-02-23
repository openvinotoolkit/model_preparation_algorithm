# model settings
_base_ = [
    '../../_base_/model.py'
]

model = dict(
    task='detection')

load_from = None
resume_from = None
