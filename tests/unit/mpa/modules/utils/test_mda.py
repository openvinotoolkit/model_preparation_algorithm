import os
# import unittest
import pytest
import shutil
from mmcv import Config
from mda import MDA

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

from mpa.utils.mda_stage import MdaRunner
from mpa.utils import mda_stage
from mpa.utils.config_utils import MPAConfig

OUTPUT_PATH = '_mpa_unittest'


def make_sample_recipe(task='classification', mda_metric='z-score'):
    if task == "detection":
        config_file = 'recipes_old/detection/domain_adapt/faster_rcnn/finetune/train.yaml'
    else:
        config_file = 'recipes_old/classification/domain_adapt/finetune/train.yaml'

    cfg = {
        'name': 'finetune',
        'task': task,
        'mode': 'train',
        'config': config_file,
        'mda_metric': mda_metric,
        'output': ['final_score'],
        'common_cfg': {'output_path': OUTPUT_PATH}
    }

    return Config(cfg)


def mock_function(*args, **kwargs):
    return 1


@pytest.fixture
def classification_model_cfg():
    res = MPAConfig.fromfile('models/classification/mobilenet_v2.yaml')
    return res


@pytest.fixture
def detection_model_cfg():
    res = MPAConfig.fromfile('models/detection/faster_rcnn_r50_fpn.py')
    res.pretrained = None
    return res


@pytest.fixture
def cifar_data_cfg():
    return MPAConfig.fromfile('tests/assets/data_cfg/cifar10split_224_bs16.py')


@pytest.fixture
def coco_data_cfg():
    return MPAConfig.fromfile('tests/assets/data_cfg/coco_smallest.py')


@pytest.fixture
def classification_recipe():
    return make_sample_recipe()


@pytest.fixture
def detection_recipe():
    return make_sample_recipe(task='detection')


@pytest.fixture
def MDA_cls():
    return MdaRunner(**make_sample_recipe(task="classification", mda_metric="z-score"))


@pytest.fixture
def MDA_det():
    return MdaRunner(**make_sample_recipe(task="detection", mda_metric="z-score"))


@pytest.fixture
def MDA_cls_cfg(MDA_cls, classification_model_cfg, cifar_data_cfg):
    MDA_cls.task_stage._init_logger()
    return MDA_cls.task_stage.configure(classification_model_cfg, None, cifar_data_cfg, training=False)


@pytest.fixture
def MDA_det_cfg(MDA_det, detection_model_cfg, coco_data_cfg):
    MDA_det.task_stage._init_logger()
    return MDA_det.task_stage.configure(detection_model_cfg, None, coco_data_cfg, training=False)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
@pytest.mark.parametrize("task", [1, 0.5, True, 'wrong_value'])
def test_MdaRunner_init_wrong_task_value(task):
    with pytest.raises(ValueError):
        MdaRunner(**make_sample_recipe(task=task))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
@pytest.mark.parametrize("mda_metric", [1, 0.5, True, 'wrong_value'])
def test_MdaRunner_init_wrong_mda_metric_value(mda_metric):
    with pytest.raises(ValueError):
        MdaRunner(**make_sample_recipe(mda_metric=mda_metric))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
@pytest.mark.parametrize("task", ["classification", "detection"])
@pytest.mark.parametrize("mda_metric", ['z-score', 'cos-sim', 'kl', 'wst'])
def test_MdaRunner_init_right_mda_metric_value(task, mda_metric):
    MdaRunner(**make_sample_recipe(task=task, mda_metric=mda_metric))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_classification_make_dataset(MDA_cls, MDA_cls_cfg):
    ds = MDA_cls.make_dataset(MDA_cls_cfg)
    assert 'mpa.modules.datasets' in str(type(ds))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
@pytest.mark.parametrize("input_source", ["train", "val", "test"])
def test_MdaRunner_detection_make_dataset(MDA_det, MDA_det_cfg, input_source):
    MDA_det_cfg.input_source = input_source
    ds = MDA_det.make_dataset(MDA_det_cfg)
    assert 'mmdet.datasets' in str(type(ds))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_classification_make_model(MDA_cls, MDA_cls_cfg):
    model = MDA_cls.make_model(MDA_cls_cfg)
    assert 'mmcls.model' in str(type(model))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_detection_make_model(MDA_det, MDA_det_cfg):
    datset = Config({'CLASSES': 10})
    MDA_det.dataset = datset
    model = MDA_det.make_model(MDA_det_cfg)
    assert 'mmdet.models' in str(type(model))


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_classification_analyze_model_drift(monkeypatch, MDA_cls, MDA_cls_cfg):
    monkeypatch.setattr(MDA, "measure", mock_function)
    monkeypatch.setattr(mda_stage, "build_dataloader_cls", mock_function)
    MDA_cls.analyze_model_drift(MDA_cls_cfg)
    assert os.path.exists(f'{OUTPUT_PATH}/stage00_finetune/mpa_output.txt')
    shutil.rmtree(OUTPUT_PATH)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_detection_analyze_model_drift(monkeypatch, MDA_det, MDA_det_cfg):
    monkeypatch.setattr(MDA, "measure", mock_function)
    monkeypatch.setattr(mda_stage, "build_dataloader_det", mock_function)
    MDA_det.analyze_model_drift(MDA_det_cfg)
    assert os.path.exists(f'{OUTPUT_PATH}/stage00_finetune/mpa_output.txt')
    shutil.rmtree(OUTPUT_PATH)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_MDA_run_wrong_mode(MDA_cls, classification_model_cfg, cifar_data_cfg):
    MDA_cls.mode = 'wrong_value'
    res = MDA_cls.run(model_cfg=classification_model_cfg, model_ckpt=None, data_cfg=cifar_data_cfg)
    assert res == {}


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_MdaRunner_run(MDA_det, detection_model_cfg, coco_data_cfg):
    MDA_det.analyze_model_drift = mock_function
    res = MDA_det.run(model_cfg=detection_model_cfg, model_ckpt=None, data_cfg=coco_data_cfg)
    assert res == 1
    assert os.path.exists(OUTPUT_PATH)
    shutil.rmtree(OUTPUT_PATH)
