import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from abc import abstractmethod

from mpa.registry import EXPLAINERS

class BaseExplainer(ABC):
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
class RISE(nn.Module):
    def __init__(self, model):
        pass

    def forward(self, x):
        pass