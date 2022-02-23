import torch
import torch.nn.functional as F

from ..builder import PIXEL_SAMPLERS
from .base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class OHEMPixelSampler(BasePixelSampler):
    """Online Hard Example Mining Sampler for segmentation.

    Args:
        context (nn.Module): The context of sampler, subclass of
            :obj:`BaseDecodeHead`.
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: None.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
    """

    def __init__(self, thresh=None, min_kept=100000, kept_ratio=None, **kwargs):
        super(OHEMPixelSampler, self).__init__(**kwargs)

        self.thresh = thresh
        self.min_kept = min_kept
        self.kept_ratio = kept_ratio

    def _sample(self, losses=None, seg_logit=None, seg_label=None, valid_mask=None):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """

        with torch.no_grad():
            if valid_mask is None:
                assert seg_label is not None
                valid_mask = seg_label != self.ignore_index

            if self.thresh is not None:
                assert seg_logit is not None
                assert seg_label is not None

                # assert seg_logit.shape[2:] == seg_label.shape[2:]
                if seg_logit.shape[2:] != seg_label.shape[2:]:
                    seg_label = seg_label.unsqueeze(dim=1)

                assert seg_label.shape[1] == 1

                seg_label = seg_label.squeeze(1).long()
                num_kept = self.min_kept * seg_label.size(0)
                seg_weight = seg_logit.new_zeros(size=seg_label.size())
                valid_seg_weight = seg_weight[valid_mask]

                seg_prob = F.softmax(seg_logit, dim=1)

                tmp_seg_label = seg_label.clone().unsqueeze(1)
                tmp_seg_label[tmp_seg_label == self.ignore_index] = 0
                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
                sort_prob, sort_indices = seg_prob[valid_mask].sort()

                if sort_prob.numel() > 0:
                    min_threshold = sort_prob[min(num_kept, sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0
                threshold = max(min_threshold, self.thresh)

                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.0
            else:
                assert losses is not None
                valid_losses = losses[valid_mask]

                if self.kept_ratio is not None:
                    assert 0.0 < self.kept_ratio < 1.0

                    num_total = valid_losses.numel()
                    num_kept = min(max(1, int(self.kept_ratio * num_total)), num_total - 1)
                else:
                    assert self.min_kept > 1
                    num_kept = self.min_kept * losses.size(0)

                seg_weight = torch.zeros_like(losses)
                valid_seg_weight = seg_weight[valid_mask]

                # faster than topk according to https://github.com/pytorch/pytorch/issues/22812  # noqa
                _, sort_indices = valid_losses.sort(descending=True)
                valid_seg_weight[sort_indices[:num_kept]] = 1.0

            seg_weight[valid_mask] = valid_seg_weight

            return seg_weight
