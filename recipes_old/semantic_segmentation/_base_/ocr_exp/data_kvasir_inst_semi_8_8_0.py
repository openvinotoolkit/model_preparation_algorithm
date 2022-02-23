[split_labels, split_unlabels, split_seed] = 8, 8, 0
_base_ = [
    f'./data_kvasir_inst_sup_{split_labels}_{split_seed}.py'
]

dataset_type = 'KvasirDataset'
data_root = './data/kvasir-instrument'
data = dict(
    train=dict(
        dataset=dict(
            type='PseudoSemanticSegDataset',
            orig_type=dataset_type,
            data_root=data_root,
            img_dir='images',
            ann_dir='masks',
            split=f'train_label_{split_labels}_seed_{split_seed}.txt',
            unlabeled_split=f'train_label_{split_labels}_unlabel_{split_unlabels}_seed_{split_seed}.txt',
        )
    )
)
