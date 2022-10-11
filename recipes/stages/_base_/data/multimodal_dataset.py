__dataset_type = 'MultimodalDataset'

__samples_per_gpu = 16

data = dict(
    samples_per_gpu=__samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type=__dataset_type,
    ),
    val=dict(
        type=__dataset_type,
    ),
    test=dict(
        type=__dataset_type,
    )
)
