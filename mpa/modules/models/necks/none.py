import torch.nn as nn

from mmcls.models.builder import NECKS


@NECKS.register_module()
class Skip(nn.Module):
    """skip
    """

    def __init__(self):
        super().__init__()

    def init_weights(self):
        pass

    def forward(self, inputs):
        return inputs