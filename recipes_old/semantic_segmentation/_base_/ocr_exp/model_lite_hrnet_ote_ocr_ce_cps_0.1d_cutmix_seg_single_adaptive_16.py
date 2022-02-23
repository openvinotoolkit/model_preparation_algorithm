# model settings
_base_ = [
    './model_lite_hrnet_ote_ocr_ce_16.py'
]

# model settings
__norm_cfg = dict(type='BN', requires_grad=True)

model = dict(type='SingleAdaptiveCutMixSegSegmentor',
             ori_type='OCRCascadeEncoderDecoder',
             cps_weight=0.1,
             backbone=dict(norm_cfg=__norm_cfg),
             )
