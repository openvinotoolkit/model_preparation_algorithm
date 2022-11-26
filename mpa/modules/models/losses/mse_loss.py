import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.pixel_base import BasePixelLoss



@LOSSES.register_module()
class MSELoss(BasePixelLoss):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def _calculate(self, pred, target, weight=None, avg_factor=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = self.loss_weight *  F.mse_loss(pred, target, reduction='none')
        return (loss, None)
    @property
    def name(self):
        return 'ce'
    
    def _pred_stat(self, output, labels, valid_mask, window_size=5, min_group_ratio=0.6):
        return None