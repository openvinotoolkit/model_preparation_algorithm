import unittest
import pytest
# from mmcls.models import BACKBONES
# from mmcls.datasets import DATASETS as CLS_DATASETS
import mpa.selfsl.builder as builder

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
class TestSelfSLBuilder(unittest.TestCase):
    def setUp(self):
        dataset_type = 'SelfSLDataset'
        down_task = 'classification'
        img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_pipeline = [
            dict(type='RandomResizedCrop', size=112),
            dict(type='ToNumpy'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
        data_cfg = dict(
            type=dataset_type,
            down_task=down_task,
            datasource=dict(
                type='TVDatasetSplit',
                data_prefix='data/torchvision/cifar10',
                base='CIFAR10',
                pipeline=[],
                train=True,
                download=True),
            pipeline=dict(
                view0=self.train_pipeline,
                view1=self.train_pipeline
            )
        )
        model_cfg = dict(
            type='SelfSL',
            donw_task=down_task,
            pretrained='torchvision://resnet50',
            backbone=dict(
                type='ResNet',
                depth=50
            ),
            neck=dict(
                type='MLP',
                in_channels=2048,
                hid_channels=256,
                out_channels=128,
                with_avg_pool=True
            ),
            head=dict(
                type='LatentPredictHead',
                loss='MSE',
                size_average=True,
                predictor=dict(
                    type='MLP',
                    in_channels=128,
                    hid_channels=256,
                    out_channels=128,
                    with_avg_pool=False
                )
            )
        )

        self.cfg = dict(
            model=model_cfg,
            data=dict(
                train=data_cfg,
            )
        )

    def tearDown(self):
        pass

    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.priority_high
    @pytest.mark.unit
    def test_build(self):
        data_cfg = self.cfg['data']['train']
        concat_data_cfg = [data_cfg, data_cfg]
        concat_dataset = builder.build_dataset(concat_data_cfg)
        dataset = builder.build_dataset(data_cfg)

        self.assertEqual(len(dataset) * 2, len(concat_dataset))
