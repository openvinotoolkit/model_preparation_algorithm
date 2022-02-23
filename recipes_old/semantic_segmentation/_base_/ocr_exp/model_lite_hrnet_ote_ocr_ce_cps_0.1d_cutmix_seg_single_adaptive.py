# # model settings
# _base_ = [
#     './model_lite_hrnet_ote_ocr_ce.py'
# ]

_base_ = [
    '/home/jihwan/mpa_common/recipes/stages/_base_/models/segmentors/segmentor.py'
]
# model settings
__norm_cfg = dict(type='BN', requires_grad=True)

model = dict(type='SingleAdaptiveCutMixSegSegmentor',
             ori_type='SemiSLSegmentor',
             cps_weight=0.1,
             backbone=dict(norm_cfg=__norm_cfg),
             )
