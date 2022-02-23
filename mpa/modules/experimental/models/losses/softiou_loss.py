"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch.nn as nn
import torch.nn.functional as F

# from ..builder import LOSSES
from mmseg.models.builder import LOSSES
from mpa.modules.models.losses.utils import weighted_loss
from mpa.modules.models.losses.utils_pixel_wise import builder
from mmseg.models.losses.cross_entropy_loss import _expand_onehot_labels


@weighted_loss
def softiou_loss(pred,
                 target,
                 valid_mask,
                 class_weight=None,
                 ignore_index=255):

    assert pred.shape[0] == target.shape[0]

    N, C, _, _ = pred.size()

    inter = pred * target
    union = pred + target - inter

    inter = inter.view(N, C, -1).sum(dim=2)
    union = union.view(N, C, -1).sum(dim=2)

    loss = inter / (union + 1e-16)

    return 1 - loss.mean()


@LOSSES.register_module()
class SoftIoULoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 **kwargs):
        super(SoftIoULoss, self).__init__()
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        # for pixel-based segmentation loss
        self._loss_weight_scheduler = builder.build_scheduler(loss_weight, default_value=1.0)

        self._iter = 0
        self._last_loss_weight = 0

        from mpa.utils import logger
        logger.info('################################ TODO ####################################')
        logger.info('# Some meta cfgs used in BasePixelLoss is not usable in SoftIoULoss yet. #')
        logger.info('# (reg_weight, scale, raw_sparsity, weight_sparsity)                     #')
        logger.info('##########################################################################')

    @property
    def iter(self):
        return self._iter

    @property
    def last_loss_weight(self):
        return self._last_loss_weight

    @property
    def name(self):
        return 'softiou'

    def _forward(self,
                 pred,
                 target,
                 avg_factor=None,
                 reduction_override=None,
                 **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        one_hot_target = _expand_onehot_labels(target, pred.shape, self.ignore_index)
        valid_mask = (target != self.ignore_index).long()

        losses = self.loss_weight * softiou_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=class_weight,
            ignore_index=self.ignore_index)

        meta = dict(
            weight=self.last_loss_weight,
            # reg_weight=self.last_reg_weight,  # TODO
            # scale=self.last_scale,            # TODO
            # raw_sparsity=raw_sparsity,        # TODO
            # weight_sparsity=weight_sparsity   # TODO
        )

        return losses, meta

    def forward(self, *args, **kwargs):
        self._last_loss_weight = self._loss_weight_scheduler(self.iter)

        loss, meta = self._forward(*args, **kwargs)
        out_loss = self._last_loss_weight * loss

        self._iter += 1

        return out_loss, meta
