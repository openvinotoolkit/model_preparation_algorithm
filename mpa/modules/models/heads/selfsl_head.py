import torch
import torch.nn as nn

from mpa.selfsl.builder import HEADS
from mpa.selfsl.builder import build_neck


def MSELoss(x, y):
    loss = 2 * x.size(0) - 2 * (x * y).sum()
    return loss / x.size(0)


def PPCLoss(q, k, coord_q, coord_k, pos_ratio):
    """ Pixel to Propagation Consistency Loss
        q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, C, H, W = q.shape
    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)
    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)

    # [bs, 1, 1]
    q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
    k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

    # [bs, 7, 7]
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

    # [bs, 49, 49]
    dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                             + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
    pos_mask = (dist_center < pos_ratio).float().detach()

    # [bs, 49, 49]
    logit = torch.bmm(q.transpose(1, 2), k)

    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()


@HEADS.register_module()
class LatentPredictHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, loss, **kwargs):
        super(LatentPredictHead, self).__init__()
        self.predictor = build_neck(predictor)
        if loss == 'MSE':
            self.regression_loss = MSELoss
        elif loss == 'PPC':
            pos_ratio = kwargs.get('pos_ratio', 0.5)
            self.regression_loss = lambda *args: PPCLoss(*args, pos_ratio=pos_ratio)
        else:
            raise ValueError(f'{loss} loss is not supported')

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, proj, proj_ng, *args):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(proj)
        pred_norm = nn.functional.normalize(pred, dim=1)
        proj_ng_norm = nn.functional.normalize(proj_ng, dim=1)

        loss = self.regression_loss(pred_norm, proj_ng_norm, *args)
        return dict(loss=loss)
