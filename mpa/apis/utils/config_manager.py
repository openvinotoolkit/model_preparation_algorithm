from typing import Optional
from sc_sdk.sc_sdk.entities.task_environment import TaskEnvironment

from mpa.apis.tasks.base import MPATaskType


class MPAConfigurationManager(object):
    def __init__(self, task_environment: TaskEnvironment, task_type: MPATaskType, scratch_space: str,
                 random_seed: Optional[int] = 42):
        """
        Class that configures an mmdetection model and training configuration. Initializes the task-specific
        configuration. Sets the work_dir for mmdetection and the number of classes in the model. Also seeds random
        generators.

        :param task_environment: Task environment for the task, containing configurable parameters, labels, etc.
        :param task_type: MMDetectionTaskType of the task at hand
        :param scratch_space: Path to working directory
        :param random_seed: Optional int to seed random generators.
        """
