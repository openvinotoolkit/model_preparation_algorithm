import torch
import torch.nn as nn
import torch.distributed as dist
from mmdet.models import LOSSES
from mpa.utils.logger import get_logger

logger = get_logger()


@LOSSES.register_module()
class EqualizedFocalLoss(nn.Module):
    """Equalized Focal Loss
    Please refer to the `paper <https://arxiv.org/abs/2201.02593>`_ for
    details.
    """

    def __init__(
        self,
        use_sigmoid=True,
        num_classes=None,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=None,
    ):
        super(EqualizedFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid

        # initial variables
        self.register_buffer("pos_grad", torch.zeros(self.num_classes))
        self.register_buffer("neg_grad", torch.zeros(self.num_classes))
        self.register_buffer("pos_neg", torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []

    def forward(
        self,
        pred,
        targets,
        weight=None,
        reduction_override=None,
        avg_factor=None,
        **kwargs
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * self.equalized_focal_loss(
            pred,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor,
        )

        return loss_cls

    def collect_grad(self, pred, target):
        # TODO[HARIM]: Upgrade collect_grad for ignore mode
        grad = target * (pred - 1) + (1 - target) * pred
        grad = grad.reshape(-1, self.num_classes)
        grad = torch.abs(grad)[self.cache_mask]

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target, dim=0)
        neg_grad = torch.sum(grad * (1 - target), dim=0)

        if torch.cuda.device_count() > 1:
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = torch.clamp(
            self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1
        )

        self.grad_buffer = []

    def equalized_focal_loss(
        self,
        inputs,
        targets,
        alpha=0.25,
        gamma=2.0,
        scale_factor=1.0,
        reduction="mean",
        avg_factor=None,
        ignore_index=-1,
    ):
        """
        Arguments:
        - inputs: inputs Tensor (N * C)
        - targets: targets Tensor (N)
        - weights: weights Tensor (N), consists of (binarized label schema * weights)
        - alpha: focal loss alpha
        - gamma: focal loss gamma
        - reduction: default = mean
        - avg_factor: average factors
        """
        n_c = inputs.shape[-1]  # C
        inputs = inputs.reshape(-1, n_c)
        targets = targets.reshape(-1)
        n_i, _ = inputs.size()  # N

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(n_i, n_c + 1)
            target[torch.arange(n_i), gt_classes] = 1
            return target[:, :-1]

        expand_target = expand_label(inputs, targets)
        sample_mask = targets != ignore_index
        inputs = inputs[sample_mask]
        targets = expand_target[sample_mask]
        self.cache_mask = sample_mask
        self.cache_target = expand_target

        pred = torch.sigmoid(inputs)
        pred_t = pred * targets + (1 - pred) * (1 - targets)

        # the accumulated gradient ratio of positive samples to negative samples per category
        map_val = 1 - self.pos_neg.detach()
        # gamma: categories-agnostic parameter
        # scale_factor * map_val: categories-specific parameter
        dy_gamma = gamma + scale_factor * map_val
        # focusing factor
        # Adopts a category-relevant focusing factor
        # to address the positive-negative imbalance of different categories separately
        # shape: (C), rare -> high
        focusing_factor = dy_gamma.view(1, -1).expand(n_i, n_c)[sample_mask]
        # weighting factor
        weighting_factor = focusing_factor / gamma

        # ce_loss
        ce_loss = -torch.log(pred_t)
        loss = (
            ce_loss
            * torch.pow((1 - pred_t), focusing_factor.detach())
            * weighting_factor.detach()
        )
        loss *= alpha

        if reduction == "mean":
            if avg_factor is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / avg_factor
        elif reduction == "sum":
            loss = loss.sum()

        # an alternate approach to obtain the gradient is by manual calculation
        self.collect_grad(pred.detach(), targets.detach())
        return loss
