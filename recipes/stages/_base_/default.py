_base_ = [
    './dist/dist.py'
]

cudnn_benchmark = True

seed = 1234
deterministic = False

hparams = dict(dummy=0)

task_adapt = dict(op='REPLACE')
