import pytest
import torch
from mpa.modules.hooks import task_adapt_hook
from mpa.modules.hooks.task_adapt_hook import TaskAdaptHook

from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def __test_task_adapt_hook(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            teacher = torch.nn.Module()
            self.teachers = [teacher]

    class Runner(object):
        def __init__(self):
            self.model = Model()

    runner = Runner()

    def fake_load(f, **kwargs):
        return dict(state_dict=dict(a=1))

    monkeypatch.setattr(torch, 'load', fake_load)

    cs_load_args = dict()

    def fake_cs_load(**kwargs):
        cs_load_args.update(**kwargs)

    monkeypatch.setattr(
        task_adapt_hook,
        'class_sensitive_copy_state_dict',
        fake_cs_load
    )

    hook = TaskAdaptHook(
        'path/to/ckpt.pth',
        ('person',),
        ('person', 'car'),
    )
    hook.before_run(runner)
    assert cs_load_args['src_dict'] == dict(a=1)
    assert cs_load_args['src_classes'] == ('person',)
    assert cs_load_args['dst_classes'] == ('person', 'car',)

    loaded_state_dict = dict()

    def fake_load_state_dict(module, state_dict, **kwargs):
        loaded_state_dict.update(state_dict)

    monkeypatch.setattr(
        task_adapt_hook,
        'load_state_dict',
        fake_load_state_dict
    )

    hook = TaskAdaptHook(
        'path/to/ckpt.pth',
        ('person',),
        ('person', 'car'),
    )
    hook.before_run(runner)
    assert loaded_state_dict == dict(a=1)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_task_adapt_hook():
    class Dataset(object):
        def __init__(self, num_images):
            self.num_images = num_images
            self.img_indices = dict()
            self.indice_setting()

        def indice_setting(self):
            indices = list(range(self.num_images))
            self.img_indices['old'] = indices[:int(self.num_images*0.8)]
            self.img_indices['new'] = indices[int(self.num_images*0.8):]

        def __len__(self):
            return self.num_images

    class DataLoader(object):
        def __init__(
            self, dataset, batch_size, num_workers,
            collate_fn=None, worker_init_fn=None
        ):
            self.dataset = dataset
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.collate_fn = collate_fn
            self.worker_init_fn = worker_init_fn

    class Runner(object):
        def __init__(self, data_loader):
            self.data_loader = data_loader

    dataset = Dataset(num_images=10)  # old 8 new 2
    data_loader = DataLoader(dataset, 2, 1)

    runner = Runner(data_loader)

    hook = TaskAdaptHook(
        ('person',),
        ('person', 'car'),
        sampler_flag=True
    )
    hook.before_epoch(runner)
    assert len(runner.data_loader.sampler.new_indices) == 2
    assert len(runner.data_loader.sampler.old_indices) == 8
    assert len(runner.data_loader.sampler) == 10
    assert runner.data_loader.sampler.old_new_ratio == 2

    # Efficient Mode
    hook = TaskAdaptHook(
        ('person',),
        ('person', 'car'),
        sampler_flag=True,
        efficient_mode=True
    )
    hook.before_epoch(runner)
    assert len(runner.data_loader.sampler.new_indices) == 2
    assert len(runner.data_loader.sampler.old_indices) == 8
    assert len(runner.data_loader.sampler) == 6
    assert runner.data_loader.sampler.old_new_ratio == 1
