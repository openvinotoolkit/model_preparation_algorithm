import torch
import torch.nn as nn
from mmcv.cnn import DepthwiseSeparableConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.losses import accuracy
from mmseg.models.builder import build_loss
from mmcv.runner import force_fp32


@HEADS.register_module()
class ReCoDepthwiseSeparableASPPHead(DepthwiseSeparableASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, reco_loss_decode=dict(type='ReCoLoss', loss_weight=1.0), **kwargs):
        super(ReCoDepthwiseSeparableASPPHead, self).__init__(**kwargs)
        c1_channels = kwargs['c1_channels']
        self.reco_loss_decode = build_loss(reco_loss_decode)
        # reco HP
        self.weak_threshold = 0.7
        self.strong_threshold = 0.97
        self.temp = 0.5
        self.num_queries = 256
        self.num_negatives = 256

        self.sep_representation = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Conv2d(self.channels, 256, kernel_size=1))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(self.image_pool(x),
                   size=x.size()[2:],
                   mode='bilinear',
                   align_corners=self.align_corners)]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        representation = self.sep_representation(output)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output, representation

    def ce_loss(self, seg_logit, gt, weight=None):
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, gt)
        else:
            if weight is None:
                seg_weight = weight
            else:
                batch_size = seg_logit.shape[0]
                valid_mask = (gt < 255).float()
                weight_view = weight.view(batch_size, -1).ge(self.strong_threshold).sum(-1)
                weighting = weight_view / valid_mask.view(batch_size, -1).sum(-1)
                # print("weighting:{}".format(weighting))
                weighting = weighting.view(batch_size, 1, 1)
                seg_weight = weighting

        seg_label = gt.squeeze(1)
        loss_seg = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index
            )
        acc_seg = accuracy(seg_logit, seg_label)

        return loss_seg, acc_seg

    def unsupervised_reco_loss(self, rep, label, mask, prob,
                               strong_threshold=1.0, temp=0.5,
                               num_queries=256, num_negatives=256):

        seg_label = label.squeeze(1)
        loss_seg = self.reco_loss_decode(rep, label, mask, prob,
                                         strong_threshold=strong_threshold, temp=temp,
                                         num_queries=num_queries, num_negatives=num_negatives)
        acc_seg = accuracy(prob, seg_label)

        return loss_seg, acc_seg

    @force_fp32(apply_to=('seg_logit', 'seg_u_logits',))
    def reco_loss(self, feature, gt_semantic_seg, mask, logits,
                  strong_threshold=0.97, temp=0.5,
                  num_queries=256, num_negatives=256):
        """Compute segmentation loss."""
        loss = dict()
        gt = gt_semantic_seg['gt']
        pseudo_label = gt_semantic_seg['pseudo_label']
        seg_logits, seg_u_logits = logits
        train_u_reco_mask, train_u_pseudo_mask = mask

        seg_logit = resize(input=seg_logits,
                           size=gt.shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        seg_u_logit = resize(input=seg_u_logits,
                             size=pseudo_label.shape[2:],
                             mode='bilinear',
                             align_corners=self.align_corners)
        loss['loss_seg'], loss['acc_seg'] = self.ce_loss(seg_logit, gt)
        loss['loss_seg_u'], loss['acc_seg_u'] = self.ce_loss(seg_u_logit, pseudo_label, train_u_pseudo_mask)
        pseudo_label = resize(input=pseudo_label.float(), size=feature.shape[2:], mode='nearest')
        train_u_reco_mask = resize(input=train_u_reco_mask.unsqueeze(1), size=feature.shape[2:], mode='nearest')
        loss['loss_seg_reco'], loss['acc_reco'] = self.unsupervised_reco_loss(feature, pseudo_label,
                                                                              train_u_reco_mask,
                                                                              seg_u_logits,
                                                                              strong_threshold=strong_threshold,
                                                                              temp=temp,
                                                                              num_queries=num_queries,
                                                                              num_negatives=num_negatives)
        # loss['loss'] = loss['loss_seg'] + loss['loss_seg_u'] + loss['loss_seg_reco']

        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        if isinstance(inputs, dict):
            x = inputs['x']
            x_u = inputs['x_u']
            conf = inputs['conf']

            seg_logits, _ = self.forward(x)
            seg_u_logits, feature = self.forward(x_u)
            logits = (seg_logits, seg_u_logits)
            train_u_reco_mask = conf.ge(self.weak_threshold).float()
            train_u_pseudo_mask = conf.ge(self.strong_threshold).float()
            mask = (train_u_reco_mask, train_u_pseudo_mask)
            losses = self.reco_loss(feature, gt_semantic_seg, mask, logits,
                                    strong_threshold=self.strong_threshold, temp=self.temp,
                                    num_queries=self.num_queries, num_negatives=self.num_negatives)
        else:
            seg_logits = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits, _ = self.forward(inputs)
        return seg_logits
