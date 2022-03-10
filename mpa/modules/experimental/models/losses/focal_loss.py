import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mpa.modules.models.losses.pixel_base import BasePixelLoss
from mmseg.models.losses.cross_entropy_loss import _expand_onehot_labels


def focal_loss(pred,
               target,
               gamma=2.0,
               alpha=0.25,
               ignore_index=255):

    pred_softmax = F.softmax(pred, 1)
    target = _expand_onehot_labels(target, pred.shape, ignore_index)

    pt = (1 - pred_softmax) * target + pred_softmax * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
    loss *= focal_weight
    return loss.mean(dim=1)


@LOSSES.register_module()
class FocalLossMPA(BasePixelLoss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super(FocalLossMPA, self).__init__(**kwargs)

        self.gamma = gamma
        self.alpha = alpha

        self.cls_criterion = focal_loss

        from mpa.utils.logger import get_logger
        logger = get_logger()
        logger.info('######################################################################################')
        logger.info('# This Focal loss is implemented based on SoftmaxFocalLoss at class_balanced_loss.py #')
        logger.info('# Differently, label smoothing is not considered.                                    #')
        logger.info('######################################################################################')

    @property
    def name(self):
        return 'focal'

    def _calculate(self, cls_score, label, scale, weight=None):
        loss = self.cls_criterion(
            scale * cls_score,
            label,
            gamma=self.gamma,
            alpha=self.alpha,
            ignore_index=self.ignore_index
        )

        return loss, cls_score
