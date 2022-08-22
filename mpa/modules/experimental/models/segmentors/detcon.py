import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init, build_norm_layer
from mmcv.runner import load_checkpoint

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from mmseg.models import builder
from mmseg.models import SEGMENTORS
from mmseg.models.segmentors import CascadeEncoderDecoder


class MaskPooling(nn.Module):
    def __init__(
        self, 
        num_classes, 
        num_samples=16, 
        downsample=32,
        replacement=False,
        ignore_bg=False
    ):

        super().__init__()

        self.num_classes = num_classes
        self.num_samples = num_samples
        self.replacement = replacement
        self.ignore_bg = ignore_bg
        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)

    def pool_masks(self, masks):
        """Create binary masks and performs mask pooling
        Args:
            masks: (b, 1, h, w)
        Returns:
            masks: (b, num_classes, d)
        """
        if masks.ndim < 4:
            masks = masks.unsqueeze(dim=1)

        masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
        masks = self.pool(masks.to(torch.float))

        b, c, h, w = masks.shape
        masks = torch.reshape(masks, (b, c, h * w))
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes).to(masks.device)[masks]
        masks = torch.transpose(masks, 1, 2)
        return masks

    def sample_masks(self, masks):
        """Samples which binary masks to use in the loss.
        Args:
            masks: [(b, num_classes, d), (b, num_classes, d)]
        Returns:
            masks: (b, num_samples, d)
        """
        bs = masks[0].shape[0]
        mask_exists = [torch.greater(mask.sum(dim=-1), 1e-3) for mask in masks]
        sel_masks = [mask_exist.to(torch.float) + 1e-11 for mask_exist in mask_exists]
        # torch.multinomial handles normalizing
        # sel_masks = sel_masks / sel_masks.sum(dim=1, keepdim=True)
        # sel_masks = torch.softmax(sel_masks, dim=-1)
        if self.ignore_bg:
            for sel_mask in sel_masks:
                sel_mask[:,0] = 0

        mask_ids = [
            torch.multinomial(sel_mask, num_samples=self.num_samples, replacement=self.replacement) 
            for sel_mask in sel_masks
        ]
        sampled_masks = [
            torch.stack([mask[b][mask_id[b]] for b in range(bs)]) 
            for mask, mask_id in zip(masks, mask_ids)
        ]
        return sampled_masks, mask_ids

    def forward(self, masks):
        """
        Args:
            masks: [mask1, mask2]
        Returns:
            sampled_masks: [sampled_mask1, sampled_mask2]
            sampled_mask_ids: [sampled_mask_ids1, sampled_mask_ids2]
        """
        binary_masks = [self.pool_masks(mask) for mask in masks]
        sampled_masks, sampled_mask_ids = self.sample_masks(binary_masks)
        areas = [sampled_mask.sum(dim=-1, keepdim=True) for sampled_mask in sampled_masks]
        sampled_masks = [
            sampled_mask / torch.maximum(area, torch.tensor(1.0, device=area.device)) 
            for sampled_mask, area in zip(sampled_masks, areas)
        ]
        return sampled_masks, sampled_mask_ids


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DetConMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d'),
                 with_avg_pool=True):
        super(DetConMLP, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            build_norm_layer(norm_cfg, hid_channels)[-1],
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x)]


@SEGMENTORS.register_module()
class DetConB(CascadeEncoderDecoder):
    def __init__(
        self,
        num_stages,
        backbone,
        decode_head,
        projector,
        predictor,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        detcon_loss_cfg=None,
        pretrained=None,
        base_momentum=0.996,
        num_classes=256,
        num_samples=16,
        downsample=32,
        input_transform=None,
        in_index=-1,
        replacement=True,
        ignore_bg=False,
        use_decode_head=True,
        use_warmstart=True,
        **kwargs
    ):

        assert use_decode_head or use_warmstart
        super(DetConB, self).__init__(
            num_stages=num_stages,
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=None)

        logger = get_root_logger()
        for i in range(self.num_stages):
            if input_transform != decode_head[i]['input_transform']:
                logger.warning(
                    f"input_transform (={input_transform}) is not matched with "
                    f"decode_head[{i}]['input_transform'] (={decode_head[i]['input_transform']}).")

            if in_index != decode_head[i]['in_index']:
                logger.warning(
                    f"in_index (={in_index}) is not matched with "
                    f"decode_head[{i}]['in_index'] (={decode_head[i]['in_index']}).")

        self.input_transform = input_transform
        self.in_index = in_index
        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.downsample = downsample
        self.use_decode_head = use_decode_head
        self.use_warmstart = use_warmstart
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample, replacement, ignore_bg)

        if pretrained:
            load_checkpoint(self.backbone, pretrained, strict=False, map_location='cpu', 
                            logger=logger, revise_keys=[(r'^backbone\.', '')])
        
        if self.use_warmstart:
            self.online_net = nn.Sequential(self.backbone, DetConMLP(**projector))
            self.target_net = nn.Sequential(builder.build_backbone(backbone), DetConMLP(**projector))
            self.predictor = DetConMLP(**predictor)

            self.online_net[1].init_weights(init_linear='kaiming') # projection
            self.predictor.init_weights()
            for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
                param_tgt.data.copy_(param_ol.data)
                param_tgt.requires_grad = False

            self.detcon_loss = builder.build_loss(detcon_loss_cfg)

        else:
            self.online_net = [self.backbone]

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tg in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tg.data = param_tg.data * self.momentum + \
                param_ol.data * (1. - self.momentum)

    def _forward_train(self, imgs, masks, net):
        embeddings = [net[0](img) for img in imgs]

        if self.use_warmstart:
            if self.input_transform:
                embeddings_for_detcon = [self._transform_inputs(embedding) for embedding in embeddings]
            elif isinstance(self.in_index, int):
                embeddings_for_detcon = [embeddings[self.in_index] for embedding in embeddings]
            else:
                raise ValueError()

            bs, emb_d, emb_h, emb_w = embeddings_for_detcon[0].shape
            sampled_masks, sampled_mask_ids = self.mask_pool(masks)

            embeddings_for_detcon = [
                embedding.reshape((bs, emb_d, emb_h*emb_w)).transpose(1, 2) for embedding in embeddings_for_detcon
            ]
            sampled_embeddings = [
                sampled_mask @ embedding_for_detcon 
                for sampled_mask, embedding_for_detcon in zip(sampled_masks, embeddings_for_detcon)
            ]
            # error -> outer product!
            # sampled_embeddings = [
            #     torch.einsum('bchw,bmd->bmc', embedding_for_detcon, sampled_mask) 
            #     for embedding_for_detcon, sampled_mask in zip(embeddings_for_detcon, sampled_masks)
            # ]

            sampled_embeddings = [
                sampled_embedding.reshape((-1, emb_d)) 
                for sampled_embedding in sampled_embeddings
            ]
            proj_outs = [net[1]([sampled_embedding])[0] for sampled_embedding in sampled_embeddings]
            return embeddings, proj_outs, sampled_mask_ids

        return embeddings, None, None

    def forward_train(self, img, img_metas, gt_semantic_seg, pixel_weights=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            aux_img (Tensor): Auxiliary images.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()
        img1, img2 = img[:, 0], img[:, 1]
        mask1, mask2 = gt_semantic_seg[:, 0], gt_semantic_seg[:, 1]
        embds, projs, ids = self._forward_train([img1, img2], [mask1, mask2], self.online_net)

        if self.use_decode_head:
            # decode head
            e1, e2 = embds
            loss_decode1, meta_decode1 = self._decode_head_forward_train(
                e1, img_metas, gt_semantic_seg=mask1, pixel_weights=pixel_weights)
            loss_decode2, meta_decode2 = self._decode_head_forward_train(
                e2, img_metas, gt_semantic_seg=mask2, pixel_weights=pixel_weights)
                
            for k, v1 in loss_decode1.items():
                v2 = loss_decode2[k]
                losses[k] = sum([v1, v2]) / 2

        if self.use_warmstart:
            with torch.no_grad():
                self._momentum_update()
                _, (ema_proj1, ema_proj2), (ema_ids1, ema_ids2) = \
                    self._forward_train([img1, img2], [mask1, mask2], self.target_net)
                
            # predictor
            proj1, proj2 = projs
            pred1, pred2 = self.predictor([proj1])[0], self.predictor([proj2])[0]
            pred1 = pred1.reshape((-1, self.num_samples, pred1.shape[-1]))
            pred2 = pred2.reshape((-1, self.num_samples, pred2.shape[-1]))
            ema_proj1 = ema_proj1.reshape((-1, self.num_samples, ema_proj1.shape[-1]))
            ema_proj2 = ema_proj2.reshape((-1, self.num_samples, ema_proj2.shape[-1]))
            
            # decon loss
            ids1, ids2 = ids
            loss_detcon = self.detcon_loss(
                pred1=pred1, 
                pred2=pred2,
                target1=ema_proj1,
                target2=ema_proj2,
                pind1=ids1,
                pind2=ids2,
                tind1=ema_ids1,
                tind2=ema_ids2)['loss']
            losses.update(dict(detcon_loss=loss_detcon))

        if self.with_auxiliary_head:
            raise ValueError()

        return losses

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


@SEGMENTORS.register_module()
class DetConSupCon(DetConB):
    def forward_train(self, img, img_metas, gt_semantic_seg, pixel_weights=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            aux_img (Tensor): Auxiliary images.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()
        img1, img2 = img[:, 0], img[:, 1]
        mask1, mask2 = gt_semantic_seg[:, 0], gt_semantic_seg[:, 1]
        embds, projs, ids = self._forward_train([img1, img2], [mask1, mask2], self.online_net)
        
        # decode head
        # In SupCon, img1 is only used for supervised learning.
        # TODO: data pipeline for SupCon
        assert self.use_decode_head
        e1, _ = embds
        loss_decode, _ = self._decode_head_forward_train(
            e1, img_metas, gt_semantic_seg=mask1, pixel_weights=pixel_weights)
            
        losses.update(loss_decode)

        # detcon head
        assert self.use_warmstart
        with torch.no_grad():
            self._momentum_update()
            _, (ema_proj1, ema_proj2), (ema_ids1, ema_ids2) = \
                self._forward_train([img1, img2], [mask1, mask2], self.target_net)
            
        # predictor
        proj1, proj2 = projs
        pred1, pred2 = self.predictor([proj1])[0], self.predictor([proj2])[0]
        pred1 = pred1.reshape((-1, self.num_samples, pred1.shape[-1]))
        pred2 = pred2.reshape((-1, self.num_samples, pred2.shape[-1]))
        ema_proj1 = ema_proj1.reshape((-1, self.num_samples, ema_proj1.shape[-1]))
        ema_proj2 = ema_proj2.reshape((-1, self.num_samples, ema_proj2.shape[-1]))
        
        # decon loss
        ids1, ids2 = ids
        loss_detcon = self.detcon_loss(
            pred1=pred1, 
            pred2=pred2,
            target1=ema_proj1,
            target2=ema_proj2,
            pind1=ids1,
            pind2=ids2,
            tind1=ema_ids1,
            tind2=ema_ids2)['loss']
        losses.update(dict(detcon_loss=loss_detcon))

        if self.with_auxiliary_head:
            raise ValueError()

        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        out, repr_vector = super().encode_decode(img, img_metas)

        return out
