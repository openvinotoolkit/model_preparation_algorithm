import io
import os
from typing import Optional

import torch

from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelPrecision #, ModelStatus
from ote_sdk.entities.resultset import ResultSetEntity

from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload

# from mmdet.apis import export_model

from mpa.apis.configs import ClassificationConfig
from mpa.apis.tasks.base import BaseTask
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger

logger = get_logger()

TASK_CONFIG = ClassificationConfig


class ClassificationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        self._should_stop = False
        super().__init__(TASK_CONFIG, task_environment)

    def infer(self,
              dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None
              ) -> DatasetEntity:
        logger.info('called infer()')

        raise NotImplementedError

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        logger.info('called evaluate()')
        raise NotImplementedError

    def unload(self):
        logger.info('called unload()')
        raise NotImplementedError

    def export(self,
               export_type: ExportType,
               output_model: ModelEntity):
        logger.info('called export()')
        raise NotImplementedError

    def _init_recipe_hparam(self) -> dict:
        return dict()

    def _init_recipe(self):
        logger.info('called _init_recipe()')

        # recipe_root = 'recipes/stages/classification'
        # train_type = self._hyperparams.algo_backend.train_type
        # logger.info(f'train type = {train_type}')

        raise NotImplementedError

    def _init_model_cfg(self, output_model: ModelEntity):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        return MPAConfig.fromfile(os.path.join(base_dir, 'model.py'))

    def _init_data_cfg(self, dataset: DatasetEntity):
        return None


class ClassificationTrainTask(ClassificationInferenceTask):
    def save_model(self, output_model: ModelEntity):
        logger.info('called save_model')
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            'model': model_ckpt, 'config': hyperparams_str, 'labels': labels,
            'confidence_threshold': self.confidence_threshold, 'VERSION': 1
        }

        # if hasattr(self._config.model, 'bbox_head') and hasattr(self._config.model.bbox_head, 'anchor_generator'):
        #     if getattr(self._config.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
        #         generator = self._model.bbox_head.anchor_generator
        #         modelinfo['anchors'] = {'heights': generator.heights, 'widths': generator.widths}

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.precision = [ModelPrecision.FP32]

    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        # self._should_stop = True
        # stop_training_filepath = os.path.join(self._training_work_dir, '.stop_training')
        # open(stop_training_filepath, 'a').close()
        if self.cancel_interface is not None:
            self.cancel_interface.cancel()
        else:
            logger.info('but training was not started yet. reserved it to cancel')
            self.reserved_cancel = True

    def train(self,
              dataset: DatasetEntity,
              output_model: ModelEntity,
              train_parameters: Optional[TrainParameters] = None):
        logger.info('train()')
        self._initialize(dataset, output_model=output_model)

        stage_module = 'ClsTrainer'
        results = self._run_task(stage_module, mode='train', parameters=train_parameters)

        # get output model
        model_ckpt = results.get('final_ckpt')
        if model_ckpt is None:
            logger.error('cannot find final checkpoint from the results.')
            # output_model.model_status = ModelStatus.FAILED
            return
        else:
            # update checkpoint to the newly trained model
            self._model_ckpt = model_ckpt

        # get prediction on validation set
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        self.infer(val_dataset)
        result_set = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=val_dataset,
            prediction_dataset=val_dataset.with_empty_annotations()
        )

        # adjust confidence threshold
        if self._hyperparams.postprocessing.result_based_confidence_threshold:
            logger.info('Adjusting the confidence threshold')
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=True)
            best_confidence_threshold = metric.best_confidence_threshold.value
            if best_confidence_threshold is None:
                raise ValueError("Cannot compute metrics: Invalid confidence threshold!")
            logger.info(f"Setting confidence threshold to {best_confidence_threshold} based on results")
            self.confidence_threshold = best_confidence_threshold
        else:
            metric = MetricsHelper.compute_f_measure(result_set, vary_confidence_threshold=False)

        # compose performance statistics
        performance = metric.get_performance()
        logger.info(f'Final model performance: {str(performance)}')
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        # output_model.model_status = ModelStatus.SUCCESS
        logger.info('train done.')

    def _init_data_cfg(self, dataset: DatasetEntity):
        pass

    def _init_recipe_hparam(self) -> dict:
        return dict(runner=dict(max_epochs=2))
