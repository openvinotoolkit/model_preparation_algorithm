import pytest
import os
import math
from unittest.mock import patch, MagicMock, Mock
from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

from mmcv.utils import Config
from mpa.modules.datasets.pseudo_balanced_dataset import PseudoBalancedDataset


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_pseudo_balanced_dataset():
    cfg = Config.fromfile('tests/assets/data_cfg/coco_smallest.yaml')
    cfg.data.train.org_type = cfg.data.train.type
    ds = PseudoBalancedDataset(
        pseudo_length=0,
        **cfg.data.train,
        pipeline=[]
    )
    assert len(ds) == 32
    ds = PseudoBalancedDataset(
        pseudo_length=-1,
        **cfg.data.train,
        pipeline=[]
    )
    assert len(ds) == int(10*math.sqrt(float(32)))
    ds = PseudoBalancedDataset(
        pseudo_length=10,
        **cfg.data.train,
        pipeline=[]
    )
    assert len(ds) == 10

