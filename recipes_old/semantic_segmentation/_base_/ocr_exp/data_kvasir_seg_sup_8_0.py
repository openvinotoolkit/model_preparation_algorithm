_base_ = [
    './data_kvasir_seg_pipeline.py'
]

[split_labels, split_unlabels, split_seed] = 8, -1, 0
data = dict(
    train=dict(
        dataset=dict(
            split=f'train_label_{split_labels}_seed_{split_seed}.txt',
        )
    )
)
