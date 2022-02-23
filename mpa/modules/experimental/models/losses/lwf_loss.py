import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import LOSSES
from mmdet.models.losses import weight_reduce_loss
from .kd_loss import KDLoss
from mpa.modules.utils.task_adapt import map_class_names


@LOSSES.register_module()
class LwFLoss(KDLoss):
    def __init__(self,
                 src_classes,
                 dst_classes,
                 bg_aware=True,
                 temperature=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        """ loss

        Args:
            src_classes (list[str]): Teacher (OLD) classes
            dst_classes (list[str]): Student (NEW) classes
            bg_aware (bool, optional): Whether to enable BG-aware distillation
                '__bg__' class would be added the end of src/dst_classes
            temperature (float, optional): Temperature for KD loss
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        print('LwFLoss init!')
        super().__init__(
            use_sigmoid=False,  # softmax only
            temperature=temperature,
            reduction=reduction,
            loss_weight=loss_weight
        )
        self.src_classes = src_classes.copy()
        self.dst_classes = dst_classes.copy()
        self.bg_aware = bg_aware
        if bg_aware:
            self.src_classes += ['__bg__']
            self.dst_classes += ['__bg__']
        self.src2dst = torch.tensor(map_class_names(self.src_classes, self.dst_classes))
        self.dst2src = torch.tensor(map_class_names(self.dst_classes, self.src_classes))

    def forward(self,
                input,
                target,
                target_is_logit=True,
                weight=None,
                avg_factor=None,
                **kwargs):
        """Forward function.

        Args:
            input (torch.Tensor): The prediction (N, C_new, ...)
            target (torch.Tensor): The learning target of the prediction (N, C_old, ...)
            weight (torch.Tensor, optional): Sample-wise loss weight (N, 1, ...)
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """

        if input.shape[0] == 0:
            return torch.tensor(0.0)

        if not self.bg_aware:
            # Gathering predictions according to class name matching
            src_pred = target[:, self.src2dst >= 0]
            dst_pred = input[:, self.src2dst[self.src2dst >= 0]]
            return super().forward(dst_pred, src_pred, weight=weight, avg_factor=avg_factor, **kwargs)

        # --- Elem-wise BG-aware LwF loss
        # Non-matching probabilities would be regarded as 'BACKGROUND' probability
        # So, they are accumulated to BG probability (both for OLD & NEW)
        if target_is_logit:
            src_prob_all = F.softmax(target/self.T, dim=1)
        else:
            src_prob_all = target
        src_prob_gathered = src_prob_all[:, self.src2dst >= 0]
        src_prob_nomatch = src_prob_all[:, self.src2dst < 0]
        src_prob_gathered[:, -1] += src_prob_nomatch.sum(dim=1)  # Accumulating to BG prob
        dst_prob_all = F.softmax(input/self.T, dim=1)
        dst_prob_gathered = dst_prob_all[:, self.src2dst[self.src2dst >= 0]]
        dst_prob_nomatch = dst_prob_all[:, self.dst2src < 0]
        dst_prob_gathered[:, -1] += dst_prob_nomatch.sum(dim=1)  # Accumulating to BG prob

        # src_logit_gathered = target[:, self.src2dst >= 0]
        # src_prob_gathered = F.softmax(src_logit_gathered/self.T, dim=1)
        # dst_logit_gathered = input[:, self.src2dst[self.src2dst >= 0]]
        # dst_prob_gathered = F.softmax(dst_logit_gathered/self.T, dim=1)

        # X-entropy
        losses = -src_prob_gathered*torch.log(dst_prob_gathered)
        # losses = losses.sum(dim=1)

        # --- Weighted reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            losses, weight=weight, reduction=self.reduction, avg_factor=avg_factor)

        return loss*self.loss_weight


@LOSSES.register_module()
class ClassSensitiveCrossEntropyLoss(nn.Module):
    def __init__(self,
                 model_classes,
                 data_classes,
                 bg_aware=True,
                 reduction='mean',
                 loss_weight=1.0):
        """ loss

        Args:
            model_classes (list[str]): Model classes
            data_classes (list[str]): Data classes
            bg_aware (bool, optional): Whether to enable BG-aware distillation
                '__bg__' class would be added the end of model/data_classes
            temperature (float, optional): Temperature for KD loss
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        print('ClassSensitiveCrossEntropyLoss init!')
        super().__init__()
        self.model_classes = model_classes.copy()
        self.data_classes = data_classes.copy()
        self.bg_aware = bg_aware
        self.reduction = reduction
        self.loss_weight = loss_weight
        if bg_aware:
            self.model_classes += ['__bg__']
            self.data_classes += ['__bg__']
        self.model2data = torch.tensor(map_class_names(self.model_classes, self.data_classes))
        # print(f'model classes = {self.model_classes}, data classes = {self.data_classes}')
        # print(f'###########  model2data mapping = {self.model2data}')

    def forward(self,
                logits,
                labels,
                weight=None,
                avg_factor=None,
                **kwargs):
        """Forward function.

        Args:
            logits (torch.Tensor): The prediction (N, C_new, ...)
            labels (torch.Tensor): The learning target of the prediction (N, ...)
            weight (torch.Tensor, optional): Sample-wise loss weight (N, 1, ...)
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """

        if logits.shape[0] == 0:
            return torch.tensor(0.0)

        # --- Elem-wise BG-aware X-entropy loss
        # Non-matching probabilities would be regarded as 'BACKGROUND' probability
        # So, they are accumulated to BG probability (both for input & target)
        prob_all = F.softmax(logits, dim=1)
        # print(f'shape of prob_all {prob_all.shape}')
        # print(f'shape of prob_all model2data {self.model2data.shape}')
        prob_gathered = prob_all[:, self.model2data >= 0]
        prob_nomatch = prob_all[:, self.model2data < 0]
        prob_gathered[:, -1] += prob_nomatch.sum(dim=1)  # Accumulating non-matching probs to BG prob
        prob_log = torch.log(prob_gathered)
        label_gathered = self.model2data[labels].to(labels)

        # X-entropy: NLL loss w/ log(softmax(logit)) & labels
        losses = F.nll_loss(prob_log, label_gathered, reduction='none')

        # logit_gathered = logits[:, self.model2data >= 0]
        # label_gathered = self.model2data[labels].to(labels)
        # logit_gathered = logits
        # label_gathered = labels
        # losses = F.cross_entropy(logit_gathered, label_gathered, reduction='none')

        # --- Weighted reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            losses, weight=weight, reduction=self.reduction, avg_factor=avg_factor)

        return loss*self.loss_weight
