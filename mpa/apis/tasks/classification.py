import signal
from typing import Optional

from sc_sdk.sc_sdk.configuration.configurable_parameters import ConfigurableParameters

from sc_sdk.sc_sdk.entities.datasets import Dataset
from sc_sdk.sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.sc_sdk.entities.model import Model

from sc_sdk.sc_sdk.logging import logger_factory

from mpa.apis.configs.classification import MPAClassificationParameters
from mpa.apis.tasks.base import MPABaseTask


logger = logger_factory.get_logger("MPATask")


class MPAClassificationTask(MPABaseTask):
    def train(self, dataset: Dataset, train_parameters: Optional[TrainParameters] = None) -> Model:
        """
        train a model on the given dataset. things below should be treated here.
        - catchs SIGTERM signal and raise a RuntimeError
        - if the target backend is GPU and multiple GPUs are available, use all of them for the training.
          multi-node training supporting is not defined yet
        - select a best checkpoint based on the validation set (early stopping?)
        """
        signal.signal(signal.SIGTERM, self._interupt_handler)

        # prepare MPA consumable config

        # build dataset

        # build model

        # do pre-eval

        # train

        # eval

        # check improvement and select best checkpoint
        raise NotImplementedError

    def save_model(self, output_model: Model):
        raise NotImplementedError

    # def optimize(self,
    #     optimization_type: OptimizationType,
    #     dataset: Optional[Dataset],
    #     output_model: OptimizedModel,
    #     optimization_parameters: Optional[OptimizationParameters] = None):
    #     raise NotImplementedError

    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> ConfigurableParameters:
        return task_environment.get_configurable_parameters(instance_of=MPAClassificationParameters)
