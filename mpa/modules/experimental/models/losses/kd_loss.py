import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import LOSSES
from mmdet.models.losses import weight_reduce_loss


@LOSSES.register_module()
class KDLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 temperature=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        """Knowledege Distillation loss

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.T = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                input,
                target,
                weight=None,
                avg_factor=None,
                **kwargs):
        """Forward function.

        Args:
            input (torch.Tensor): The prediction (N, C, ...)
            target (torch.Tensor): The learning target of the prediction (N, C, ...)
            weight (torch.Tensor, optional): Sample-wise loss weight (N, 1, ...)
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """

        # Elem-wise KD loss
        if self.use_sigmoid:
            input_prob = torch.sigmoid(input/self.T)
            target_prob = torch.sigmoid(target/self.T)
            losses = (target_prob - input_prob)**2
        else:
            input_log_prob = F.log_softmax(input/self.T, dim=1)
            target_prob = F.softmax(target/self.T, dim=1)
            losses = (-target_prob*input_log_prob).sum(dim=1)

        # losses = F.kl_div(input_log_prob, target_prob, reduction='none')

        # Weighted reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            losses, weight=weight, reduction=self.reduction, avg_factor=avg_factor)

        return loss*self.loss_weight
