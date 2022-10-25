import torch.nn as nn

from abc import ABC
from abc import abstractmethod

from mpa.registry import EXPLAINERS


class BaseExplainer(ABC, nn.Module):
    """
    Blackbox explainer base class
    """
    def __init__(self, model):
        self._model = model
    
    @abstractmethod
    def get_saliency_map(self):
        pass

    @abstractmethod
    def run(self):
        pass

# RISE: Randomized Input Sampling for Explanation of Black-box Models
# https://arxiv.org/pdf/1806.07421.pdf

@EXPLAINERS.register_module()
class RISE(BaseExplainer):
    pass


@EXPLAINERS.register_module()
class A_RISE(BaseExplainer):
    pass


@EXPLAINERS.register_module()
class D_RISE(BaseExplainer):
    pass