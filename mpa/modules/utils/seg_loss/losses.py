import torch
import torch.nn.functional as F
import numpy as np


def entropy(p, dim=1, keepdim=False):
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


def focal_loss(input_values, gamma):
    return (1.0 - torch.exp(-input_values)) ** gamma * input_values


class MaxEntropyLoss:
    def __init__(self, scale=1.0):
        super(MaxEntropyLoss, self).__init__()

        self.scale = scale
        assert self.scale > 0.0

    def forward(self, cos_theta):
        probs = F.softmax(self.scale * cos_theta, dim=1)

        entropy_values = entropy(probs, dim=1)
        losses = np.log(cos_theta.size(-1)) - entropy_values

        return losses.mean()


class CrossEntropy:
    def __init__(self, weight=1.0, class_weight=None):
        self.weight = weight
        self.class_weight = class_weight

    def __call__(self, logits, target, weight=None, class_weight=None):
        if self.class_weight is not None:
            class_weight = logits.new_tensor(self.class_weight)

        return self.weight * F.cross_entropy(logits, target, weight=class_weight, reduction='none')


class NormalizedCrossEntropy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, logits, target):
        log_softmax = F.log_softmax(logits, dim=1)
        b, c, h, w = log_softmax.size()

        log_softmax = log_softmax.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.view(-1)

        target_log_softmax = log_softmax[torch.arange(target.size(0), device=target.device), target]
        target_log_softmax = target_log_softmax.view(b, h, w)

        sum_log_softmax = log_softmax.sum(dim=1)
        losses = self.weight * target_log_softmax / sum_log_softmax

        return losses


class ReverseCrossEntropy:
    def __init__(self, scale=4.0, weight=1.0):
        self.weight = weight * abs(float(scale))

    def __call__(self, logits, target):
        all_probs = F.softmax(logits, dim=1)
        b, c, h, w = all_probs.size()

        all_probs = all_probs.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.view(-1)

        target_probs = all_probs[torch.arange(target.size(0), device=target.device), target]
        target_probs = target_probs.view(b, h, w)

        losses = self.weight * (1.0 - target_probs)

        return losses


class SymmetricCrossEntropy:
    def __init__(self, alpha=1.0, beta=1.0):
        self.ce = CrossEntropy(weight=alpha)
        self.rce = ReverseCrossEntropy(weight=beta)

    def __call__(self, logits, target):
        return self.ce(logits, target) + self.rce(logits, target)


class ActivePassiveLoss:
    def __init__(self, alpha=100.0, beta=1.0):
        self.active_loss = NormalizedCrossEntropy(weight=alpha)
        self.passive_loss = ReverseCrossEntropy(weight=beta)

    def __call__(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)


class EqualizationLossV2:
    def __init__(self, gamma=12, mu=0.8, alpha=4.0):
        from tl.modules.models.losses.eqlv2 import EQLv2
        self.loss = EQLv2(gamma=gamma, mu=mu, alpha=alpha)

    def __call__(self, logits, target):
        losses, _ = self.loss._calculate(logits, target, 1)
        return losses


def build_classification_loss(name, class_weight=None):
    if name == 'ce':
        return CrossEntropy(class_weight=class_weight)
    elif name == 'nce':
        return NormalizedCrossEntropy()
    elif name == 'rce':
        return ReverseCrossEntropy()
    elif name == 'sl':
        return SymmetricCrossEntropy()
    elif name == 'apl':
        return ActivePassiveLoss()
    elif name == 'eqlv2':
        return EqualizationLossV2()
    else:
        raise AttributeError('Unknown name of loss: {}'.format(name))
