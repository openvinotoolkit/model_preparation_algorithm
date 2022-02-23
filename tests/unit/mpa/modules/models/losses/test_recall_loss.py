import pytest
import torch

from mpa.modules.models.losses.recall_loss import RecallLoss

from mmseg.models.builder import build_loss

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_recall_loss():
    loss_cfg = dict(
        type='RecallLoss',
        loss_jitter_prob=0.01,
        loss_weight=1.0
    )

    recall_loss = build_loss(loss_cfg)

    assert isinstance(recall_loss, RecallLoss)
    assert recall_loss.name == 'recall_loss'

    cls_score = torch.randn((1, 4, 32, 32))
    label = torch.randint(low=0, high=4, size=(1, 32, 32)).long()

    loss, _ = recall_loss._calculate(cls_score, label, 1.)

    assert loss.shape == label.shape
