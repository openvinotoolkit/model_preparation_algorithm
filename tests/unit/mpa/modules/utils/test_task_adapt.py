import pytest
import torch
from mpa.modules.utils.task_adapt import map_class_names, class_sensitive_copy_state_dict, extract_anchor_ratio,\
                                         map_cat_and_cls_as_order, prob_extractor

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements
from unittest.mock import patch, MagicMock


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_map_class_names():
    src_classes = ['person', 'car', 'tree']
    dst_classes = ['tree', 'person', 'sky', 'ball']
    gt_src2dst = [1, -1, 0]
    src2dst = map_class_names(src_classes, dst_classes)
    assert src2dst == gt_src2dst

    src_classes = ['person', 'car', 'tree']
    dst_classes = []
    gt_src2dst = [-1, -1, -1]
    src2dst = map_class_names(src_classes, dst_classes)
    assert src2dst == gt_src2dst

    src_classes = []
    dst_classes = ['tree', 'person', 'sky', 'ball']
    gt_src2dst = []
    src2dst = map_class_names(src_classes, dst_classes)
    assert src2dst == gt_src2dst


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_class_sensitive_copy_state_dict():
    # with pytest.raises(NotImplementedError):  # Changed to warning
    class_sensitive_copy_state_dict({}, [], {}, [], 'SomeNewDetector')


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_class_sensitive_copy_state_dict_cls():
    import os
    assets_path = 'tests/assets'
    model_path = os.path.join(assets_path, 'model_cfg/task_inc/parameters.pth')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
    class_sensitive_copy_state_dict(state_dict, ['class1'], state_dict, ['class1', 'class2'], 'TaskIncrementalLwF')
    wrong_params = os.path.join(assets_path, 'model_cfg/ckpt/cifar10_5cls_mnet_v2.pth')
    wrong_state_dict = torch.load(wrong_params, map_location=torch.device('cpu'))['state_dict']
    logger_warning = MagicMock()
    with patch('mpa.utils.logger.warning', logger_warning):
        class_sensitive_copy_state_dict(wrong_state_dict, ['class1'], state_dict, ['class1', 'class2'],
                                        'TaskIncrementalLwF')
    logger_warning.assert_called()


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_extract_anchor_ratio():
    from mmdet.datasets.builder import build_dataset
    input_size = 512
    img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
    data_cfg = dict(
        samples_per_gpu=30,
        workers_per_gpu=3,
        train=dict(
            type='CocoDataset',
            classes=['bird'],
            ann_file='data/coco/annotations/semi_supervised/instances_train2017.1@1.0.json',
            min_size=20,
            img_prefix='data/coco/train2017',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.1),
                dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        ),
    )
    dataset = build_dataset(data_cfg['train'])
    res = extract_anchor_ratio(dataset)
    assert len(res) == 5
    res = extract_anchor_ratio(dataset, 7)
    assert len(res) == 7


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_map_cat_and_cls_as_order():
    classes = ['car', 'person']
    cats = {0: {'id': 0, 'name': 'person'}, 1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}}
    cat2labels, cat_ids = map_cat_and_cls_as_order(classes, cats)
    assert len(cat2labels) == 2
    assert len(cat_ids) == 2


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_prob_extractor():
    from mmcls.models import build_classifier
    model_cfg = dict(
        type='TaskIncrementalLwF',
        pretrained=None,
        backbone=dict(
            type='MobileNetV2',
            widen_factor=1.0,
        ),
        neck=dict(
            type='GlobalAveragePooling'
        ),
        head=dict(
            type='TaskIncLwfHead',
            in_channels=1280,
            tasks=dict(
                Age=["Other", "Senior", "Kids", "Unknown"]
            ),
            distillation_loss=dict(
                type='LwfLoss',
                T=2
            ),
            topk=(1,)
        )
    )
    data_loader = []
    for i in range(4):
        dummy_image = torch.rand(3, 3, 224, 224)
        data_loader.append(dict(img=dummy_image))
    model = build_classifier(model_cfg)
    old_prob, _ = prob_extractor(model, data_loader)
    assert 'Age' in old_prob
    assert old_prob['Age'].shape == (12, 4)
