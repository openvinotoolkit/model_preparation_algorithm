import torch

from ..builder import PIXEL_SAMPLERS
from .base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class ClassWeightingPixelSampler(BasePixelSampler):
    def __init__(self, context, eps=1e-5):
        super().__init__(context)

        self.eps = eps

    def _sample(self, losses=None, seg_logit=None, seg_label=None, valid_mask=None):
        with torch.no_grad():
            assert seg_logit is not None
            assert seg_label is not None
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1

            seg_label = seg_label.view(-1)
            class_ids, reverse_ind, class_sizes = torch.unique(
                seg_label, return_inverse=True, return_counts=True
            )

            invalid_mask = class_ids == self.context.ignore_index

            class_weights = self._get_weights(class_sizes, invalid_mask, self.eps)
            seg_weight = class_weights[reverse_ind]

            logit_shape = seg_logit.size()
            seg_weight = seg_weight.view((logit_shape[0],) + logit_shape[2:])

            return seg_weight

    @staticmethod
    def _get_weights(class_counts, invalid_mask, eps):
        frequencies = torch.reciprocal(class_counts.float()) + eps
        frequencies[invalid_mask] = 0.0

        weights = frequencies / (torch.sum(frequencies))

        return weights
