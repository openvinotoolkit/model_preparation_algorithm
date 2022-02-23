_base_ = [
    './train.py',
    '../_base_/models/detectors/detector.py'
]

task_adapt = dict(
    type='mpa',
    op='MERGE',
    efficient_mode=False,
)
