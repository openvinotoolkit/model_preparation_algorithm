_base_ = '../model.py'

task = 'detection'

model = dict(
    train_cfg=dict(),
    test_cfg=dict()
)

checkpoint_config = dict(interval=5, max_keep_ckpts=1)
