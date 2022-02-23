import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init
from mmcv.cnn import build_norm_layer

from mpa.selfsl.builder import NECKS


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


@NECKS.register_module()
class MLP(nn.Module):
    """The MLP neck: fc/conv-bn-relu-fc/conv.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d'),
                 use_conv=False,
                 with_avg_pool=True):
        super(MLP, self).__init__()

        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_conv = use_conv
        if use_conv:
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, 1),
                build_norm_layer(norm_cfg, hid_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_channels, out_channels, 1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                build_norm_layer(norm_cfg, hid_channels)[1],
                nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels)
            )

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            # using last output
            x = x[-1]
        if not isinstance(x, torch.Tensor):
            raise TypeError('neck inputs should be tuple or torch.tensor')
        if self.with_avg_pool:
            x = self.avgpool(x)
        if self.use_conv:
            return self.mlp(x)
        else:
            return self.mlp(x.view(x.size(0), -1))


@NECKS.register_module()
class PPM(nn.Module):
    """
    Head for Pixpro
    """
    def __init__(self, sharpness, **kwargs):
        super(PPM, self).__init__()
        self.transform = nn.Conv2d(256, 256, 1)
        self.sharpness = sharpness

    def init_weights(self, **kwargs):
        pass

    def forward(self, feat):
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if not isinstance(feat, torch.Tensor):
            raise TypeError('neck inputs should be tuple or torch.tensor')

        N, C, H, W = feat.shape

        # Value transformation
        feat_value = self.transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)

        # Similarity calculation
        feat = F.normalize(feat, dim=1)

        # [N, C, H * W]
        feat = feat.view(N, C, -1)

        # [N, H * W, H * W]
        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=0.)
        attention = attention ** self.sharpness

        # [N, C, H * W]
        feat = torch.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, H, W)
