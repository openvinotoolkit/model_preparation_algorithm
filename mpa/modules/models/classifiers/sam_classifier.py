from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models.classifiers.image import ImageClassifier


@CLASSIFIERS.register_module()
class SAMClassifier(BaseClassifier):
    """SAM-enabled BaseClassifier"""

    def train_step(self, data, optimizer):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer)


@CLASSIFIERS.register_module()
class SAMImageClassifier(ImageClassifier):
    """SAM-enabled ImageClassifier"""

    def train_step(self, data, optimizer):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer)
