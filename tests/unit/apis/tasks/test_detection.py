import time
from unittest.mock import Mock, patch

import pytest
from mpa.apis.configs.base import TrainType
from mpa.apis.tasks.detection import DetectionTrainTask
from mpa.modules.hooks.cancel_interface_hook import CancelInterfaceHook
from mpa.utils.logger import get_logger
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.resultset import ResultSetEntity
# from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from tests.constants.mpa_components import MPAComponent
from tests.constants.requirements import Requirements

logger = get_logger()


def task_worker_with_assert_called_for_patch_obj(
        task,
        task_attr,              # task attribute to be called
        task_args_dict,
        mock_obj=None,
        patched_target=None,    # patched target
        patched_attr=None,      # patched object
        max_epoch=5
        ):
    mock_init_recipe_hparam = Mock(return_value=dict(runner=dict(max_epochs=max_epoch)))
    with patch.object(DetectionTrainTask, '_init_recipe_hparam', mock_init_recipe_hparam):
        if mock_obj is not None:
            with patch.object(patched_target, patched_attr, mock_obj):
                getattr(task, task_attr)(**task_args_dict)
            mock_obj.assert_called()
        else:
            getattr(task, task_attr)(**task_args_dict)


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_detection_task_train_progress_callback(
        fixture_task_env,
        fixture_dataset_entity,
        fixture_train_parameters,
        ):
    task = DetectionTrainTask(fixture_task_env)

    output_model = ModelEntity(
        fixture_dataset_entity,
        fixture_task_env.get_model_configuration(),
        # model_status=ModelStatus.NOT_READY
    )

    task_args_dict = dict(
            dataset=fixture_dataset_entity,
            output_model=output_model,
    )

    # test with empty return of _run_task()
    mock_obj = Mock(return_value=dict())
    task_worker_with_assert_called_for_patch_obj(
        task,
        'train',
        task_args_dict,
        mock_obj,
        DetectionTrainTask,
        '_run_task'
    )
    # assert output_model.model_status == ModelStatus.FAILED

    # test with progress_callback
    pcallback = Mock()
    fixture_train_parameters.update_progress = pcallback

    task_args_dict['train_parameters'] = fixture_train_parameters

    task_worker_with_assert_called_for_patch_obj(
        task,
        'train',
        task_args_dict,
        max_epoch=1
    )
    pcallback.assert_called()


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_detection_task_cancel_train(
        fixture_task_env,
        fixture_dataset_entity,
        fixture_train_parameters
        ):

    import threading

    task = DetectionTrainTask(fixture_task_env)

    output_model = ModelEntity(
        fixture_dataset_entity,
        fixture_task_env.get_model_configuration(),
        # model_status=ModelStatus.NOT_READY
    )

    mock_obj = Mock()
    # check calling of cancel_training()
    task_args_dict = dict(
            dataset=fixture_dataset_entity,
            output_model=output_model,
    )
    # run through the worker thread
    logger.info('\t\t[[[ start worker thread ]]]')
    worker = threading.Thread(
        target=task_worker_with_assert_called_for_patch_obj,
        args=(
            task,
            'train',
            task_args_dict,
            mock_obj,
            CancelInterfaceHook,
            'cancel',
        )
    )
    worker.start()
    logger.info('\t\t[[[ worker started and wait for starting the task (sleep 3) ]]]')
    # wait for starting the task
    time.sleep(3)
    logger.info('\t\t[[[ request cancel to worker with mock object ]]]')
    task.cancel_training()
    logger.info('\t\t[[[ joining thread... ]]]')
    worker.join(timeout=5)
    # worker should be alived since join() has to be returned with timeout
    # because cancel() interface was patched as a mock and will not do something to
    # actual cancel the task.
    assert worker.is_alive()
    logger.info('\t\t[[[ timeout while joining thread ]]]')
    worker.join()   # wait for actul termination
    logger.info('\t\t[[[ joined thread ]]]')

    # actual cancel training
    worker = threading.Thread(
        target=task_worker_with_assert_called_for_patch_obj,
        args=(
            task,
            'train',
            task_args_dict,
        )
    )
    worker.start()
    logger.info('\t\t[[[ worker started and wait for starting the task (sleep 3) ]]]')
    # wait for starting the task
    time.sleep(3)
    logger.info('\t\t[[[ request cancel to worker with mock object ]]]')
    task.cancel_training()
    logger.info('\t\t[[[ joining thread... ]]]')
    worker.join(60)
    # worker should not be alived since the task should be terminated by cancel
    assert not worker.is_alive()
    logger.info('\t\t[[[ joined thread ]]]')


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_detection_task_train_export_and_save(
        fixture_task_env,
        fixture_dataset_entity,
        fixture_train_parameters
        ):
    task = DetectionTrainTask(fixture_task_env)

    output_model = ModelEntity(
        fixture_dataset_entity,
        fixture_task_env.get_model_configuration(),
        # model_status=ModelStatus.NOT_READY
    )

    task_args_dict = dict(
            dataset=fixture_dataset_entity,
            output_model=output_model,
    )
    task_worker_with_assert_called_for_patch_obj(
        task,
        'train',
        task_args_dict,
        max_epoch=1)

    # for testing _load_model() and infer() from the task_environment
    fixture_task_env.model = output_model
    new_task = DetectionTrainTask(fixture_task_env)

    val_dataset = fixture_dataset_entity.get_subset(Subset.VALIDATION)
    predicted_val_dataset = new_task.infer(
        val_dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True))

    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=val_dataset,
        prediction_dataset=predicted_val_dataset,
    )
    new_task.evaluate(resultset)

    # train
    task_worker_with_assert_called_for_patch_obj(
        new_task,
        'train',
        task_args_dict,
        max_epoch=1)

    exported_model = ModelEntity(
        fixture_dataset_entity,
        fixture_task_env.get_model_configuration(),
        # model_status=ModelStatus.NOT_READY
    )
    new_task.export(ExportType.OPENVINO, exported_model)
    # assert exported_model.model_status == ModelStatus.SUCCESS


@pytest.mark.components(MPAComponent.MPA)
@pytest.mark.reqids(Requirements.REQ_1)
@pytest.mark.priority_high
@pytest.mark.unit
def test_detection_task_hyperparams(
        fixture_task_env,
        fixture_dataset_entity,
        fixture_train_parameters
        ):

    output_model = ModelEntity(
        fixture_dataset_entity,
        fixture_task_env.get_model_configuration(),
        # model_status=ModelStatus.NOT_READY
    )

    # not implemented train types
    hyper_parameters = fixture_task_env.get_hyper_parameters()
    hyper_parameters.algo_backend.train_type = TrainType.FutureWork
    task = DetectionTrainTask(fixture_task_env)
    with pytest.raises(Exception) as e:
        task.train(fixture_dataset_entity, output_model)
    assert e.type == NotImplementedError

    hyper_parameters = fixture_task_env.get_hyper_parameters()
    hyper_parameters.algo_backend.train_type = TrainType.SelfSupervised
    task = DetectionTrainTask(fixture_task_env)
    with pytest.raises(Exception) as e:
        task.train(fixture_dataset_entity, output_model)
    assert e.type == NotImplementedError

    hyper_parameters = fixture_task_env.get_hyper_parameters()
    hyper_parameters.algo_backend.train_type = TrainType.DynamicLabels
    task = DetectionTrainTask(fixture_task_env)
    with pytest.raises(Exception) as e:
        task.train(fixture_dataset_entity, output_model)
    assert e.type == NotImplementedError

    hyper_parameters = fixture_task_env.get_hyper_parameters()
    hyper_parameters.algo_backend.train_type = TrainType.FutureWork
    task = DetectionTrainTask(fixture_task_env)
    with pytest.raises(Exception) as e:
        task.train(fixture_dataset_entity, output_model)
    assert e.type == NotImplementedError
