_base_ = [
    '../_base_/models/segmentors/seg_ocr-lite-hrnet-18-mod2.py',
    '../../../../../../mmsegmentation/submodule/configs/_base_/datasets/pascal_voc12.py',
    '../../../../../../mmsegmentation/submodule/configs/_base_/default_runtime.py',
    '../../../../../../mmsegmentation/submodule/configs/_base_/schedules/schedule_cos_40k.py'
]

task = 'segmentation'

task_adapt = None

seed = 42
deterministic = True
find_unused_parameters = False
