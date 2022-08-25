_base_ = [
    '../../../../../../mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2/model.py',
]

task = 'segmentation'

task_adapt = None

seed = 42
deterministic = True
