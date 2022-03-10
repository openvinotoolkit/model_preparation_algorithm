import abc
import io
import os
import shutil
import tempfile
from typing import Optional, Union

import torch
from mmcv.utils.config import Config, ConfigDict
from mpa.builder import build
from mpa.modules.hooks.cancel_interface_hook import CancelInterfaceHook
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelPrecision
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, UpdateProgressCallback

logger = get_logger()


class _MPAUpdateProgressCallbackWrapper(UpdateProgressCallback):
    """ UpdateProgressCallback wrapper
        just wrapping the callback instance and provides error free representation as 'pretty_text'
    """
    def __init__(self, callback, **kwargs):
        if not callable(callback):
            raise RuntimeError(f'cannot accept a not callable object!! {callbacl}')
        self._callback = callback
        super().__init__(**kwargs)

    def __repr__(self):
        return f"'{__name__}._MPAUpdateProgressCallbackWrapper'"

    def __reduce__(self):
        return (self.__class__, (id(self),))

    def __call__(self, progress: float, score: Optional[float] = None):
        self._callback(progress, score)


class BaseTask:
    def __init__(self, task_config, task_environment: TaskEnvironment):
        self._task_config = task_config
        self._task_environment = task_environment
        self._hyperparams = task_environment.get_hyper_parameters(self._task_config)
        self._model_name = task_environment.model_template.name
        self._labels = task_environment.get_labels(include_empty=False)
        self._output_path = tempfile.mkdtemp(prefix='MPA-task-')
        logger.info(f'created output path at {self._output_path}')
        self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        # Set default model attributes.
        self._optimization_methods = []
        self._precision = [ModelPrecision.FP32]
        self._model_ckpt = None
        if task_environment.model is not None:
            logger.info('loading the model from the task env.')
            state_dict = self._load_model_state_dict(self._task_environment.model)
            self._model_ckpt = os.path.join(self._output_path, 'env_model_ckpt.pth')
            if os.path.exists(self._model_ckpt):
                os.remove(self._model_ckpt)
            torch.save(state_dict, self._model_ckpt)

        # property below will be initialized by initialize()
        self._recipe_cfg = None
        self._stage_module = None
        self._model_cfg = None
        self._data_cfg = None
        self._mode = None
        self.cancel_interface = None
        self.reserved_cancel = False
        self.on_hook_initialized = self.OnHookInitialized(self)

    def _run_task(self, stage_module, mode=None, dataset=None, parameters=None, **kwargs):
        self._initialize(dataset)
        logger.info(f'running task... kwargs = {kwargs}')
        if self._recipe_cfg is None:
            raise RuntimeError(
                "'recipe_cfg' is not initialized yet."
                "call prepare() method before calling this method")
        # self._stage_module = stage_module
        self._mode = mode
        if parameters is not None:
            if isinstance(parameters, TrainParameters):
                hook_name = 'TrainProgressUpdateHook'
                progress_callback = _MPAUpdateProgressCallbackWrapper(parameters.update_progress)
                # TODO: update recipe to do RESUME
                if parameters.resume:
                    pass
            elif isinstance(parameters, InferenceParameters):
                hook_name = 'InferenceProgressUpdateHook'
                progress_callback = _MPAUpdateProgressCallbackWrapper(parameters.update_progress)
        else:
            hook_name = 'ProgressUpdateHook'
            progress_callback = None
        logger.info(f'progress callback = {progress_callback}, hook name = {hook_name}')
        if progress_callback is not None:
            progress_update_hook_cfg = ConfigDict(
                type='ProgressUpdateHook',
                name=hook_name,
                callback=progress_callback
            )
            update_or_add_custom_hook(self._recipe_cfg, progress_update_hook_cfg)

        common_cfg = ConfigDict(dict(output_path=self._output_path))

        # build workflow using recipe configuration
        workflow = build(self._recipe_cfg, self._mode, stage_type=stage_module, common_cfg=common_cfg)

        # run workflow with task specific model config and data config
        output = workflow.run(
            model_cfg=self._model_cfg,
            data_cfg=self._data_cfg,
            ir_path=None,
            model_ckpt=self._model_ckpt,
            mode=self._mode,
            **kwargs
        )
        logger.info('run task done.')
        return output

    def finalize(self):
        if self._recipe_cfg is not None:
            if self._recipe_cfg.get('cleanup_outputs', False):
                if os.path.exists(self._output_path):
                    shutil.rmtree(self._output_path, ignore_errors=False)

    def __del__(self):
        self.finalize()

    def _pre_task_run(self):
        pass

    @property
    def model_name(self):
        return self._task_environment.model_template.name

    @property
    def labels(self):
        return self._task_environment.get_labels(False)

    @property
    def template_file_path(self):
        return self._task_environment.model_template.model_template_path

    @property
    def hyperparams(self):
        return self._hyperparams

    def _initialize(self, dataset, output_model=None):
        """ prepare configurations to run a task through MPA's stage
        """
        logger.info('initializing....')
        self._init_recipe()
        recipe_hparams = self._init_recipe_hparam()
        if len(recipe_hparams) > 0:
            self._recipe_cfg.merge_from_dict(recipe_hparams)

        # prepare model config
        self._model_cfg = self._init_model_cfg()

        # add Cancel tranining hook
        update_or_add_custom_hook(self._recipe_cfg, ConfigDict(
            type='CancelInterfaceHook', init_callback=self.on_hook_initialized))

        logger.info('initialized.')

    @abc.abstractmethod
    def _init_recipe(self):
        """
        initialize the MPA's target recipe. (inclusive of stage type)
        """
        raise NotImplementedError('this method should be implemented')

    def _init_model_cfg(self) -> Union[Config, None]:
        """
        initialize model_cfg for override recipe's model configuration.
        it can be None. (MPA's workflow consumable)
        """
        return None

    def _init_train_data_cfg(self, dataset: DatasetEntity) -> Union[Config, None]:
        """
        initialize data_cfg for override recipe's data configuration.
        it can be Config or None. (MPA's workflow consumable)
        """
        return None

    def _init_test_data_cfg(self, dataset: DatasetEntity) -> Union[Config, None]:
        """
        initialize data_cfg for override recipe's data configuration.
        it can be Config or None. (MPA's workflow consumable)
        """
        return None

    def _init_recipe_hparam(self) -> dict:
        """
        initialize recipe hyperparamter as dict.
        """
        return dict()

    def _load_model_state_dict(self, model: ModelEntity):
        # If a model has been trained and saved for the task already, create empty model and load weights here
        buffer = io.BytesIO(model.get_data("weights.pth"))
        model_data = torch.load(buffer, map_location=torch.device('cpu'))

        # set confidence_threshold as well
        self.confidence_threshold = model_data.get('confidence_threshold', self.confidence_threshold)

        return model_data['model']

    def cancel_hook_initialized(self, cancel_interface: CancelInterfaceHook):
        logger.info('cancel hook is initialized')
        self.cancel_interface = cancel_interface
        if self.reserved_cancel:
            self.cancel_interface.cancel()

    class OnHookInitialized:
        def __init__(self, task_instance):
            self.task_instance = task_instance

        def __call__(self, cancel_interface):
            self.task_instance.cancel_hook_initialized(cancel_interface)

        def __repr__(self):
            return f"'{__name__}.OnHookInitialized'"

        def __reduce__(self):
            return (self.__class__, (id(self.task_instance),))
