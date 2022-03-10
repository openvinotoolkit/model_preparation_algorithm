import torch
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial

from mmseg.models.builder import LOSSES
from mpa.modules.models.losses.pixel_base import BasePixelLoss
from mmseg.models.losses.cross_entropy_loss import _expand_onehot_labels
from mpa.utils.logger import get_logger

logger = get_logger()


@LOSSES.register_module()
class EQLv2(BasePixelLoss):
    def __init__(self,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 **kwargs):

        super(EQLv2, self).__init__(**kwargs)
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    @property
    def name(self):
        return 'eqlv2'

    def _calculate(self, cls_score, label, scale, weight=None):
        cls_score *= scale
        self.n_i, self.n_c, self.h, self.w = cls_score.size()

        target = _expand_onehot_labels(label, cls_score.size(), self.ignore_index)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        # (16, 21, 512, 512) -> (16, 21, 512, 512)
        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target.float(), reduction='none')
        cls_loss = torch.mean(cls_loss * weight, dim=1)  # (N, 512, 512)

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return cls_loss, cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)

        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        if torch.cuda.device_count() > 1:
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad

        self.pos_neg = self._pos_grad.mean(dim=(1, 2)) / (self._neg_grad.mean(dim=(1, 2)) + 1e-10)  # (21)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros((self.n_c, self.h, self.w))
            self._neg_grad = cls_score.new_zeros((self.n_c, self.h, self.w))
            neg_w = cls_score.new_ones((self.n_i, self.n_c, self.h, self.w))
            pos_w = cls_score.new_ones((self.n_i, self.n_c, self.h, self.w))
        else:
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)

            neg_w = neg_w.view(1, self.n_c, 1, 1).expand(self.n_i, self.n_c, self.h, self.w)
            pos_w = pos_w.view(1, self.n_c, 1, 1).expand(self.n_i, self.n_c, self.h, self.w)

        return pos_w, neg_w
