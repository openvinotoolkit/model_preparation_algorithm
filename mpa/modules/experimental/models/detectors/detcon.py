from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmdet.models import builder
from mmdet.models.builder import DETECTORS
from mpa.modules.models.detectors.sam_detector_mixin import SAMDetectorMixin
from mpa.modules.models.detectors.l2sp_detector_mixin import L2SPDetectorMixin
from mmdet.models.detectors.single_stage import SingleStageDetector
from mpa.utils.logger import get_logger


class MaskPooling(nn.Module):
    def __init__(
        self, 
        num_classes, 
        num_samples=16, 
        downsample=32,
        ignore_bg=False,
        ignore_overlap=0.
    ):

        super().__init__()

        self.num_classes = num_classes
        self.num_samples = num_samples
        self.ignore_bg = ignore_bg
        self.ignore_overlap = ignore_overlap

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
        overlapped_region = None
        if self.ignore_overlap > 0:
            overlapped_region = (masks.sum(dim=1) >= self.ignore_overlap).to(torch.float)
            overlapped_region = torch.reshape(overlapped_region, (b, 1, h * w))

        masks = torch.reshape(masks, (b, c, h * w)) # (b, c, h * w)
        masks = torch.argmax(masks, dim=1) # (b, h * w)
        masks = torch.eye(self.num_classes).to(masks.device)[masks] # (b, h * w, num_classes)
        masks = torch.transpose(masks, 1, 2) # (b, num_classes, h * w)
        return masks, overlapped_region

    def sample_masks(self, masks, overlapped_region):
        """Samples which binary masks to use in the loss.
        Args:
            masks: [(b, num_classes, d), (b, num_classes, d)]
        Returns:
            masks: (b, num_samples, d)
        """
        bs = masks[0].shape[0]
        mask_exists = [torch.greater(mask.sum(dim=-1), 1e-3) for mask in masks] # (b, num_classes)
        sel_masks = [mask_exist.to(torch.float) + 1e-11 for mask_exist in mask_exists] # (b, num_classes)
        # torch.multinomial handles normalizing
        # sel_masks = sel_masks / sel_masks.sum(dim=1, keepdim=True)
        # sel_masks = torch.softmax(sel_masks, dim=-1)
        if self.ignore_bg:
            for sel_mask in sel_masks:
                sel_mask[:,0] = 0

        if self.ignore_overlap > 0:
            masks = [m * o for m, o in zip(masks, overlapped_region)]

        mask_ids = [
            torch.multinomial(sel_mask, num_samples=self.num_samples, replacement=True) 
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
        sampled_masks, sampled_mask_ids = self.sample_masks(*list(zip(*binary_masks)))
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


@DETECTORS.register_module()
class DetConB(SAMDetectorMixin, L2SPDetectorMixin, SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        projector=None,
        predictor=None,
        detcon_loss_cfg=None,
        base_momentum=0.996,
        num_classes=256,
        num_samples=16,
        downsample=32,
        input_transform=None,
        in_index=None,
        ignore_bg=False,
        ignore_overlap=0.,
        distance_overlap=False,
        loss_weights=None,
        pretrained=None,
        **kwargs
    ):

        assert projector and predictor
        assert not (ignore_overlap > 0 and distance_overlap), \
            'For test, setting both at the same time is temporarily prohibited.'

        super(DetConB, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']

        self.logger = get_logger()
        self.input_transform = input_transform
        self.in_index = in_index
        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.ignore_overlap = ignore_overlap
        self.distance_overlap = distance_overlap
        self.loss_weights = loss_weights

        self.num_classes = num_classes
        self.num_samples = num_samples
        self.downsample = downsample
        self.mask_pool = MaskPooling(num_classes, num_samples, downsample, ignore_bg, ignore_overlap)

        if pretrained is not None:
            self.logger.info('load model from: {}'.format(pretrained))
            load_checkpoint(self.backbone, pretrained, strict=False, map_location='cpu',
                            logger=self.logger, revise_keys=[(r'^backbone\.', '')])
        else:
            self.logger.warn((
                '`pretrained` is None. If `load_from` is set, '
                'weights of online network and target network will not be matched!'))
        
        self.online_net = self.backbone
        self.projector_online = DetConMLP(**projector)

        self.target_net = deepcopy(self.online_net)
        self.projector_target = DetConMLP(**projector)
        self.predictor = DetConMLP(**predictor)

        self.projector_online.init_weights(init_linear='kaiming') # projection
        self.predictor.init_weights()
        for param_ol, param_tgt in zip(self.online_net.parameters(), 
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

        for param_ol, param_tgt in zip(self.projector_online.parameters(), 
                                       self.projector_target.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

        self.detcon_loss = builder.build_loss(detcon_loss_cfg)
        
    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tg in zip(self.online_net.parameters(),
                                      self.target_net.parameters()):
            param_tg.data = param_tg.data * self.momentum + \
                param_ol.data * (1. - self.momentum)

        for param_ol, param_tg in zip(self.projector_online.parameters(),
                                      self.projector_target.parameters()):
            param_tg.data = param_tg.data * self.momentum + \
                param_ol.data * (1. - self.momentum)

    def get_transformed_features(self, x):
        if self.input_transform:
            return self._transform_inputs(x)
        elif isinstance(self.in_index, int):
            return x[self.in_index]
        elif self.decode_head:
            return self.decode_head._transform_inputs(x)
        else:
            raise ValueError()

    def _forward_train(self, imgs, masks, net, projector):
        embds = [net(img) for img in imgs]
        
        embds_for_detcon = [self.get_transformed_features(emb) for emb in embds]

        bs, emb_d, emb_h, emb_w = embds_for_detcon[0].shape
        sampled_masks, sampled_mask_ids = self.mask_pool(masks)

        embds_for_detcon = [
            embd.reshape((bs, emb_d, emb_h*emb_w)).transpose(1, 2) for embd in embds_for_detcon
        ]

        sampled_embds = [
            sampled_mask @ embd_for_detcon 
            for sampled_mask, embd_for_detcon in zip(sampled_masks, embds_for_detcon)
        ]

        sampled_embds = [sampled_embd.reshape((-1, emb_d)) for sampled_embd in sampled_embds]

        proj_outs = [projector([sampled_embd])[0] for sampled_embd in sampled_embds]

        return embds, proj_outs, sampled_mask_ids

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
        _, projs, ids = self._forward_train([img1, img2], [mask1, mask2], self.online_net, self.projector_online)

        with torch.no_grad():
            self._momentum_update()
            _, (ema_proj1, ema_proj2), (ema_ids1, ema_ids2) = \
                self._forward_train([img1, img2], [mask1, mask2], self.target_net, self.projector_target)
            
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
                F.interpolate(
                    x, 
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False
                ) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


@DETECTORS.register_module()
class DetConBSupCon(DetConB):
    def _convert_bboxes_to_masks(self, masks, bboxes, labels):
        """
        Convert annotations: bboxes to masks
        """
        if self.distance_overlap:
            distance_matrixs = None

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = map(lambda x: x.to(torch.int), bbox)

            if self.distance_overlap:
                center = ((x1+(x2-1))/2, (y1+(y2-1))/2)

                # get distance matrix
                xs = torch.linspace(x1, x2-1, x2-x1, device=x1.device)
                ys = torch.linspace(y1, y2-1, y2-y1, device=y1.device)
                Y, X = torch.meshgrid(ys-center[1], xs-center[0])

                distance_matrix = torch.zeros_like(masks, dtype=masks.dtype) + float('inf')
                distance_matrix[y1:y2,x1:x2] = Y**2 + X**2
                if distance_matrixs is None:
                    distance_matrixs = distance_matrix.unsqueeze(0)
                else:
                    distance_matrixs = torch.cat((distance_matrixs, distance_matrix.unsqueeze(0)), dim=0)

                # assign new labels based on the distances from the center of each bbox
                if torch.count_nonzero(masks[y1:y2,x1:x2]): # check if there is overlapped region
                    nonzero_coord = tuple(map(lambda x: x[0]+x[1], zip(torch.where(masks[y1:y2,x1:x2] != 0), [y1, x1])))
                    masks[nonzero_coord] = (labels[distance_matrixs.argmin(dim=0)[nonzero_coord]] + 1).to(masks.dtype)
                    masks[y1:y2,x1:x2] = torch.where(masks[y1:y2,x1:x2] == 0, (label + 1).to(masks.dtype), masks[y1:y2,x1:x2])
                else:
                    masks[y1:y2,x1:x2] = label + 1
            
            elif self.ignore_overlap > 0:
                # not assigned / same class -> label + 1
                # different class -> -1
                masks[y1:y2,x1:x2] = torch.where(
                    (masks[y1:y2,x1:x2] == 0) | (masks[y1:y2,x1:x2] == label + 1),
                    label + 1, -1)

            else:
                masks[y1:y2,x1:x2] = label + 1

        return masks

    def _prepare_masks_from_boxes(self, img1, img2, bboxes1, bboxes2, labels1, labels2):
        """
        Generates mask images by using bbox information
        """
        bs, _, h, w = img1.shape
        
        mask_canvas1 = torch.zeros((bs, 1, h, w), dtype=torch.int8, device=img1.device)
        mask_canvas2 = torch.zeros((bs, 1, h, w), dtype=torch.int8, device=img2.device)
        
        for b in range(bs):
            b_bboxes1, b_bboxes2 = bboxes1[b], bboxes2[b]
            b_labels1, b_labels2 = labels1[b], labels2[b]
            mask_canvas1[b, 0, ...] = self._convert_bboxes_to_masks(mask_canvas1[b, 0, ...], b_bboxes1, b_labels1)
            mask_canvas2[b, 0, ...] = self._convert_bboxes_to_masks(mask_canvas2[b, 0, ...], b_bboxes2, b_labels2)
        
        return mask_canvas1, mask_canvas2

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        losses = dict()

        if not isinstance(img, (list, tuple)):
            # supervised learning with interval
            e1 = self.online_net(img)

            # bbox head
            loss_bbox = self.bbox_head.forward_train(
                self.neck(e1), img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs)

            losses.update(loss_bbox)

        else:
            # supcon training
            img1, img2 = img[0], img[1]
            img_metas1, _ = img_metas[0], img_metas[1]
            gt_bboxes1, gt_bboxes2 = gt_bboxes[0], gt_bboxes[1]
            gt_labels1, gt_labels2 = gt_labels[0], gt_labels[1]

            mask1, mask2 = self._prepare_masks_from_boxes(img1, img2, gt_bboxes1, gt_bboxes2, gt_labels1, gt_labels2)

            embds, projs, ids = self._forward_train([img1, img2], [mask1, mask2], self.online_net, self.projector_online)
            
            # bbox head
            e1, _ = embds
            loss_bbox = self.bbox_head.forward_train(self.neck(e1), img_metas1, gt_bboxes1, gt_labels1, gt_bboxes_ignore=None, **kwargs)
            losses.update(loss_bbox)

            with torch.no_grad():
                self._momentum_update()
                _, (ema_proj1, ema_proj2), (ema_ids1, ema_ids2) = \
                    self._forward_train([img1, img2], [mask1, mask2], self.target_net, self.projector_target)
                
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
            losses.update(dict(detcon=loss_detcon))

            if self.loss_weights:
                reweights = dict()
                if isinstance(self.loss_weights, (list, tuple)):
                    # TODO: refactoring
                    # If `loss_weights` is changed through cli and includes `decode.*`,
                    # all of params of loss_weights is included in new `loss_weights`.
                    # It also should be List and converting it to dict is required only first time.
                    # ex) hyperparams.model.loss_weights=\"['decode.loss_seg', 0.1, 'detcon', 1]\" in cli
                    #     --> self.loss_weights = {'decode.loss_seg': 0.1, 'detcon': 1}
                    self.loss_weights = {
                        self.loss_weights[i*2]: self.loss_weights[i*2+1] 
                        for i in range(len(self.loss_weights)//2)
                    }

                for k, v in self.loss_weights.items():
                    if k not in losses:
                        self.logger.warning(
                            f'\'{k}\' is not in losses. Please check if \'loss_weights\' is set well.')
                        continue
                        
                    if 'loss_' in k:
                        reweights.update({k: losses[k] * v})
                    else:
                        reweights.update({'loss_'+k: losses[k] * v})

                losses.update(reweights)

            else:
                losses.update(dict(loss_detcon=loss_detcon))

        return losses
