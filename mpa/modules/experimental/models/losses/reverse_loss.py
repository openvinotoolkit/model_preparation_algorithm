import torch
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mpa.modules.models.losses.utils import get_class_weight
from mpa.modules.models.losses.pixel_base import BasePixelLoss


def reverse_loss(pred,
                 label,
                 weight=None,
                 class_weight=None,
                 reduction='mean',
                 avg_factor=None,
                 ignore_index=255):
    """The wrapper function for :func:`F.cross_entropy`"""

    if len(label.size()) <= 3:
        loss = F.cross_entropy(
            pred,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=255)
        # print('sup!')
    else:
        student_logit = pred
        teacher_logit = label

        student_softmax_logit = student_logit.softmax(dim=1).clone().detach()
        student_pseudo_confidence, student_pseudo_label = torch.max(student_softmax_logit, dim=1)

        teacher_softmax_logit = teacher_logit.softmax(dim=1).clone().detach()
        teacher_pseudo_confidence, teacher_pseudo_label = torch.max(teacher_softmax_logit, dim=1)

        # self correction loss is only applied to pseudo_labels.
        probabilities_current = student_softmax_logit.gather(dim=1, index=teacher_pseudo_label.unsqueeze(1)).squeeze(1)
        w = torch.max(student_softmax_logit, dim=1)[0]

        first = (-1*(w * teacher_pseudo_confidence.to(dtype=w.dtype) * torch.log(probabilities_current)))
        second = (-1*(1-w) * probabilities_current * torch.log(teacher_pseudo_confidence.to(dtype=w.dtype)))

        loss = torch.mean(first) + torch.mean(second)
        # print('unsup!')

    return loss


def _expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels


def binary_cross_entropy(pred,
                         label,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """

    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label = _expand_onehot_labels(label, pred.shape, ignore_index)

    loss = F.binary_cross_entropy_with_logits(
        pred,
        label.float(),
        pos_weight=class_weight,
        reduction='none'
    )

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """

    assert ignore_index is None, 'BCE loss does not support ignore_index'

    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    loss = F.binary_cross_entropy_with_logits(
        pred_slice,
        target,
        weight=class_weight,
        reduction='mean'
    )[None]

    return loss


@LOSSES.register_module()
class ReverseLearningLoss(BasePixelLoss):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 **kwargs):
        super(ReverseLearningLoss, self).__init__(**kwargs)

        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.class_weight = get_class_weight(class_weight)

        assert (use_sigmoid is False) or (use_mask is False)
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = reverse_loss

    @property
    def name(self):
        return 'rev'

    def _calculate(self, cls_score, label, scale, weight=None):
        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        loss = self.cls_criterion(
            scale * cls_score,
            label,
            weight,
            class_weight=class_weight,
            ignore_index=self.ignore_index
        )

        return loss, cls_score
