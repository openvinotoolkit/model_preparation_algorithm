import torch
from mmdet.ops.nms import batched_nms


def pseudo_multiclass_nms(multi_bboxes,
                          multi_scores,
                          score_thr,
                          nms_cfg,
                          max_num=-1,
                          score_factors=None):
    """NMS for multi-class bboxes w/ class probability output

    Coped from mmdet/core/post_processing/bbox_nms.py
    to augment NMS output w/ class probabilities as pseudo labels
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    # ----------------- Attach corresponding class probs as pseudo labels
    valid_prob_indices = valid_mask.nonzero(as_tuple=False)[:, 0]  # Prob row indices w/ any valid score
    valid_probs = multi_scores[valid_prob_indices]  # Gathering valid class probs
    class_probs = valid_probs[keep]  # Again gathering non-suppressed class probs
    dets = torch.cat((dets, class_probs), dim=1)
    # -----------------

    return dets, labels[keep]
