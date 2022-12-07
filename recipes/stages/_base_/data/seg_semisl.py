_base_ = [
    './data_seg.py',
    './pipelines/seg_semisl.py'
]

__dataset_type = 'KvasirDataset'
__data_root = 'data/Kvasir-SEG'
__unlabeled_pipeline = {{_base_.unlabeled_pipeline}}

data = dict(
    unlabeled=dict(
        type=__dataset_type,
        data_root=__data_root,
        pipeline=__unlabeled_pipeline,
        # cutmix=True
    )
)
