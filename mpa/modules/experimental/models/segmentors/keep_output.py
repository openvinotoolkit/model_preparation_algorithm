from mmseg.ops import resize
from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor


@SEGMENTORS.register_module()
class KeepOutputsWrapper(BaseSegmentor):
    def __init__(self, orig_type, **kwargs):
        super(KeepOutputsWrapper, self).__init__()

        cfg = kwargs.copy()
        cfg['type'] = orig_type
        self.segmentor = build_segmentor(cfg)

        if orig_type == 'OCRCascadeEncoderDecoder':
            cfg['type'] = 'OCRCascadeEncoderDecoder'
            self.align_corners = cfg['decode_head'][-1].align_corners
        else:
            cfg['type'] = 'EncoderDecoder'
            self.align_corners = cfg['decode_head'].align_corners

        print('####################### KeepOutputsWrapper')

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        segmentor = self.segmentor

        x = segmentor.extract_feat(img)
        losses = segmentor._decode_head_forward_train(x, img_metas, gt_semantic_seg)

        logits = segmentor._decode_head_forward_test(x, img_metas)
        losses['logits'] = resize(input=logits,
                                  size=img.shape[2:],
                                  mode='bilinear',
                                  align_corners=self.align_corners)
        losses['gt_semantic_seg'] = gt_semantic_seg.float()

        return losses

    def simple_test(self, img, img_meta, **kwargs):
        return self.segmentor.simple_test(img, img_meta, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))
        outputs['logits'] = losses['logits']
        outputs['gt_semantic_seg'] = losses['gt_semantic_seg'].long()
        return outputs
