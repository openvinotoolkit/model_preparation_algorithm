# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.functional import nll_loss, log_softmax

from mmcls.models.builder import LOSSES


@LOSSES.register_module()
class TwistLoss(nn.Module):
    """
    TWIST loss: https://arxiv.org/abs/2110.07402
    Self-Supervised Learning by Estimating Twin Class Distributions
    Code adapted from https://github.com/bytedance/TWIST.
    """
    def __init__(self, lam1=0.0, lam2=1.0, eps=1e-5, tau=1.0, loss_weight=1.0):
        super(TwistLoss, self).__init__()
        self.loss_weight = loss_weight
        self.lam1 = lam1
        self.lam2 = lam2
        self.eps = eps
        self.tau = tau

    def forward(self, features, labels=None, fc_feats=None):
        """
        Compute the TWIST loss and, if labels are not none, also the
        Cross-Entropy loss.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            fc_feats: tensor to train the linear classifier on
        Returns:
            A dictionary containing the loss in the 'loss' key.
        """

        losses = dict()
        losses['loss'] = 0

        # Cross-Entropy loss: classification loss
        if fc_feats is not None and labels is not None:
            if fc_feats.shape[0] == labels.shape[0] * 2:
                losses['loss'] = nll_loss(log_softmax(fc_feats, dim=1), torch.cat([labels, labels], dim=0))
            else:
                losses['loss'] = nll_loss(log_softmax(fc_feats, dim=1), labels)

            losses['loss'] *= self.loss_weight

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        feat1 = features[:, 0, :]
        feat2 = features[:, 1, :]
        probs1 = torch.nn.functional.softmax(feat1, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2, dim=-1)
        loss = dict()
        loss['kl'] = 0.5 * (KL(probs1, probs2, self.eps) + KL(probs2, probs1, self.eps))

        sharpened_probs1 = torch.nn.functional.softmax(feat1/self.tau, dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2/self.tau, dim=-1)
        loss['eh'] = 0.5 * (EH(sharpened_probs1, self.eps) + EH(sharpened_probs2, self.eps))

        # whether use historical data
        loss['he'] = 0.5 * (HE(sharpened_probs1, self.eps) + HE(sharpened_probs2, self.eps))

        loss['final'] = loss['kl'] + ((1+self.lam1)*loss['eh'] - self.lam2*loss['he'])
        losses['loss'] += self.loss_weight * loss['final']
        return losses


def KL(probs1, probs2, eps):
    kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    return kl


def CE(probs1, probs2, eps):
    ce = - (probs1 * (probs2 + eps).log()).sum(dim=1)
    ce = ce.mean()
    return ce


def HE(probs, eps):
    mean = probs.mean(dim=0)
    ent  = - (mean * (mean + eps).log()).sum()
    return ent


def EH(probs, eps):
    ent = - (probs * (probs + eps).log()).sum(dim=1)
    mean = ent.mean()
    return mean
