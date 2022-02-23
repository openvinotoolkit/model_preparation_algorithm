import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss


def lwf_loss(pred, label, T=2.0, weight=None, reduction='mean', avg_factor=None):
    # # element-wise losses
    # label = F.softmax(torch.from_numpy(label/T).cuda(), dim=1)
    label = F.softmax(label / T, dim=1)
    loss = -torch.sum(F.log_softmax(pred/T, 1) * label)
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class LwfLoss(nn.Module):

    def __init__(self,
                 T=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(LwfLoss, self).__init__()
        self.T = T
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = lwf_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.criterion(
            cls_score,
            label,
            self.T,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
