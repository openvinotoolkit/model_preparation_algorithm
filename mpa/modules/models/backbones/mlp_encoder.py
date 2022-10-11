import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init
from mmcv.cnn import build_norm_layer

from mpa.multimodal.builder import BACKBONES


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    if init_linear not in ['normal', 'kaiming']:
        raise ValueError("Undefined init_linear: {}".format(init_linear))
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

@BACKBONES.register_module()
class MLPEncoder(nn.Module):
    """ The MLP Encoder for tabular data
    """
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d'),
                 drop_out=0.2):
    
        super(MLPEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            build_norm_layer(norm_cfg, hidden_channels)[1],
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_channels, out_channels),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(),
            nn.Dropout(drop_out)
        )
        
    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        try:
            return self.mlp(x)
        except:
            raise RuntimeError(f'x.shape: {x.shape}, mlp: {self.mlp}')