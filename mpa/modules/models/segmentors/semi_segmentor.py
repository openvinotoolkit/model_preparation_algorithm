from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder, SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models import build_loss


@SEGMENTORS.register_module()
class SemiSLSegmentor(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.
    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ori_type=None):
        self.num_stages = num_stages
        super(SemiSLSegmentor, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas, self.test_cfg)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        out = self.decode_head.forward_test(x, img_metas, self.test_cfg)

        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""

        losses = dict()

        loss_decode, _ = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        return losses

    def _get_consistency_loss(self, x, img_metas, gt_semantic_seg):

        losses = dict()
        org_loss = self.decode_head.loss_modules
        self.decode_head.loss_modules = nn.ModuleList([
            build_loss(dict(type='MSELoss'))
        ])
        
        loss_decode, _ = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))
        self.decode_head.loss_modules = org_loss
        
        return losses

