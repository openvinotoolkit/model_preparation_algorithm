import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

from mmseg.models.builder import HEADS


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
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


@HEADS.register_module
class RegionCLNonLinearHeadV1(nn.Module):
    """The non-linear head in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(RegionCLNonLinearHeadV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x, randStartW=None, randStartH=None, randWidth=None, randHeight=None, randperm=None, unShuffle=None):
        assert len(x) == 1
        x = x[0]
        if randStartW is None:
            if self.with_avg_pool:
                x = self.avgpool(x)
            return self.mlp(x.view(x.size(0), -1))
        else:
            mask = torch.ones_like(x, device=x.device)
            if randHeight is None:
                randHeight = randWidth
            mix_mean_shuffle = torch.mean(x[:, :, randStartH:randStartH+randHeight, randStartW:randStartW+randWidth], [2, 3])
            mask[:, :, randStartH:randStartH+randHeight, randStartW:randStartW+randWidth] = 0.
            origin_mean = torch.sum(x * mask, dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
            feature = torch.cat([origin_mean, mix_mean_shuffle], 0)
            origin_mean, mix_mean_shuffle = torch.chunk(self.mlp(feature), 2)
            origin_mean_shuffle = origin_mean[randperm].clone()
            mix_mean = mix_mean_shuffle[unShuffle].clone()
            return origin_mean, origin_mean_shuffle, mix_mean, mix_mean_shuffle
