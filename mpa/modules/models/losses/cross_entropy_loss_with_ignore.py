# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F

from mmseg.core import focal_loss
from mmseg.models.builder import LOSSES
from .mpa_pixel_base import MPABasePixelLoss
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss
from mpa.modules.utils.task_adapt import map_class_names


@LOSSES.register_module()
class CrossEntropyLossWithIgnore(MPABasePixelLoss):
    """CrossEntropyLossWithIgnore with Ignore Mode Support for Class Incremental Learning.

    Args:
        model_classes (list[str]): Model classes
        bg_aware (bool, optional): Whether to enable BG-aware loss
            'background' class would be added the start of model classes/label schema
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 model_classes,
                 bg_aware=True,
                 reduction='mean',
                 loss_weight=None,
                 **kwargs):
        super(CrossEntropyLossWithIgnore, self).__init__(**kwargs)

        self.model_classes = model_classes.copy()
        self.bg_aware = bg_aware
        self.reduction = reduction
        self.class_weight = get_class_weight(loss_weight)
        if bg_aware:
            model_classes_without_bg = [c for c in self.model_classes if c != 'background']
            self.model_classes = ['background'] + model_classes_without_bg

    @property
    def name(self):
        return 'ce_with_ignore'

    def _calculate(self, cls_score, label, scale):
        if cls_score.shape[0] == 0:
            return torch.tensor(0.0)

        import numpy as np
        model2data_batch = []
        batch_size = label.shape[0]
        for i in range(batch_size):
            gt = np.unique(label[i])
            label_schema = []
            for idx in gt:
                label_schema.append(self.model_classes[idx])
            model2data = map_class_names(self.model_classes, label_schema)
            model2data_batch.append(model2data)
        model2data = torch.tensor(model2data_batch)
        label = torch.from_numpy(label).to(cls_score.device)

        probs_all = F.softmax(scale * cls_score, dim=1)
        losses_l = []
        for i in range(batch_size):
            probs_gathered = probs_all[i, model2data[i] >= 0]
            probs_nomatch = probs_all[i, model2data[i] < 0]
            probs_gathered = torch.unsqueeze(probs_gathered, 0)
            probs_nomatch = torch.unsqueeze(probs_nomatch, 0)

            probs_gathered[:, 0] += probs_nomatch.sum(dim=1)
            each_prob_log = torch.log(probs_gathered)

            # X-entropy: NLL loss w/ log-probabilities & labels
            each_label = torch.unsqueeze(label[i], 0)
            each_label = model2data[i][each_label]
            each_label = each_label.to(cls_score.device)
            loss = F.nll_loss(each_prob_log, each_label, reduction='none', ignore_index=self.ignore_index)
            losses_l.append(loss)

        losses = torch.cat(losses_l, dim=0)

        return losses, cls_score
