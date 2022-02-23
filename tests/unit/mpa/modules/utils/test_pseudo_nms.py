import pytest
import torch

from mpa.modules.experimental.utils.pseudo_nms import pseudo_multiclass_nms

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_pseudo_multiclass_nms(monkeypatch):

    # num_classes = 1
    score_thr = 0.8
    max_num = 1
    multi_bboxes = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
    ])
    multi_scores = torch.tensor([
        [0.1, 0.9]
    ])
    # gt_bboxes = torch.zeros((0, 5))
    # gt_labels = torch.zeros((0,), dtype=torch.long)
    bboxes, labels = pseudo_multiclass_nms(
        multi_bboxes, multi_scores, score_thr, {}
    )
    assert bboxes.numel() == 0
    assert labels.numel() == 0

    multi_bboxes = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
    ])
    multi_scores = torch.tensor([
        [0.9, 0.1]
    ])
    # gt_bboxes = torch.tensor([
    #     [1.0, 2.0, 3.0, 4.0, 0.9, 0.9, 0.1],
    #     # [x, y, w, h, max_conf, conf_0, ..., conf_num_classes]
    # ])
    # gt_labels = torch.tensor([
    #     0
    # ])

    def fake_nms(bboxes, scores, labels, nms_cfg):
        return torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9]]), torch.tensor([True])
    from mpa.modules.experimental.utils import pseudo_nms
    monkeypatch.setattr(pseudo_nms, 'batched_nms', fake_nms)
    bboxes, labels = pseudo_multiclass_nms(
        multi_bboxes, multi_scores, score_thr, {}, max_num
    )
