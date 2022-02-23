import pytest
import numpy as np

from mpa.modules.datasets.task_adapt_dataset import AdaptClassLabels
from mpa.modules.datasets.task_adapt_dataset import TaskAdaptEvalDataset

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_task_adapt_eval_dataset(monkeypatch):

    dataset = TaskAdaptEvalDataset(
        model_classes=('car',),
        org_type='CocoDataset',
        classes=('person', 'car',),
        ann_file='tests/assets/data_cfg/instances_train2017.1@0.01.json',
        pipeline=[],
    )

    def fake_eval(results, **kwargs):
        return results
    monkeypatch.setattr(dataset.dataset, 'evaluate', fake_eval)

    org_det = [[
        [np.array([1.0, 2.0, 3.0, 4.0, 1.0])]  # car
    ]]
    gt_adapt_det = [[
        [np.empty([0, 5])],                    # person
        [np.array([1.0, 2.0, 3.0, 4.0, 1.0])]  # car
    ]]
    adapt_det = dataset.evaluate(org_det)
    print(adapt_det)

    assert(len(adapt_det[0]) == 2)
    assert((adapt_det[0][1][0] == gt_adapt_det[0][1][0]).all())


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_adapt_class_labels():
    op = AdaptClassLabels(['car', 'person'], ['person', 'bus', 'car'])
    data = dict(gt_labels=[0, 1, 0, 1, 0])
    data = op(data)
    assert((data['gt_labels'] == [2, 0, 2, 0, 2]).all())
