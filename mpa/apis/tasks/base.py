from typing import Optional

from sc_sdk.sc_sdk.configuration.configurable_parameters import ConfigurableParameters

from sc_sdk.sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.sc_sdk.entities.datasets import Dataset
from sc_sdk.sc_sdk.entities.metrics import Performance
from sc_sdk.sc_sdk.entities.model import Model
from sc_sdk.sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.sc_sdk.entities.resultset import ResultSetEntity

from sc_sdk.sc_sdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from sc_sdk.sc_sdk.usecases.tasks.interfaces.configurable_parameters_interface import IConfigurableParameters
from sc_sdk.sc_sdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
from sc_sdk.sc_sdk.usecases.tasks.interfaces.unload_interface import IUnload

from sc_sdk.sc_sdk.logging import logger_factory
from ote_sdk.configuration import ConfigurableEnum

logger = logger_factory.get_logger("MPATask")


class MPATaskType(ConfigurableEnum):
    CLS_FINETUNE = 'ClassificationFinetune'
    CLS_CLASS_INCR = 'ClassificationClassIncremental'
    CLS_TASK_INCR = 'ClassificationTaskIncremental'
    CLS_SELFSL = 'ClassificationSelfSL'
    CLS_SEMISL = 'ClassificationSemiSL'

    DET_FINETUNE = 'DetectionFinetune'
    DET_SELFSL = 'DetectionSelfSL'
    DET_SEMISL = 'DetectionSemiSL'


class MPABaseTask(ImageDeepLearningTask, IConfigurableParameters, IModelOptimizer, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        """
        A Task implementation for classification fine-tune using MPA
        """
        # self.model_rev_entity = task_environment.model
        # self.hyperparams = task_environment.hyperparams
        # self.label_schema_entity = task_environment.label_schema
        self.task_environment = task_environment

    # implementations of ITask
    def analyse(self, dataset: Dataset, analyse_parameters: Optional[AnalyseParameters] = None) -> Dataset:
        raise NotImplementedError

    # implementations of IComputesPerformance
    def compute_performance(self, resultset: ResultSetEntity) -> Performance:
        raise NotImplementedError

    # implementations of ITrainInterface
    def save_model(self, output_model: Model):
        raise NotImplementedError

    def train(self, dataset: Dataset, train_parameters: Optional[TrainParameters] = None) -> Model:
        """
        train a model on the given dataset. things below should be treated here.
        - catchs SIGTERM signal and raise a RuntimeError
        - if the target backend is GPU and multiple GPUs are available, use all of them for the training.
          multi-node training supporting is not defined yet
        - select a best checkpoint based on the validation set (early stopping?)
        """
        raise NotImplementedError

    def _interupt_handler(self, signal, frame):
        logger.warning(f'catched signal {signal}')
        # TODO: save current model?
        raise RuntimeError(f'recived {signal}')

    # implementations of IConfiguableParameters
    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> ConfigurableParameters:
        raise NotImplementedError

    def update_configurable_parameters(self, task_environment: TaskEnvironment):
        raise NotImplementedError
