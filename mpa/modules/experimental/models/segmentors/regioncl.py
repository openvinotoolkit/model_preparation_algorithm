import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mmcv.runner import load_checkpoint

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from mmseg.models import builder
from mmseg.models import SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder


@SEGMENTORS.register_module
class RegionCLM(EncoderDecoder):
    """RegionCLM.

    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Part of the code is borrowed from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        head=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        regioncl_loss_cfg=None,
        pretrained=None,
        queue_len=65536,
        feat_dim=128,
        momentum=0.999,
        cutMixUpper=4,
        cutMixLower=1,
        downsampling=32.,
        in_index=-1,
        input_transform=None,
        align_corners=False,
        **kwargs
    ):

        super(RegionCLM, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=None)

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']

        self.logger = get_root_logger()
        self.downsampling = downsampling
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners

        self.backbone = self.encoder_q = builder.build_backbone(backbone)
        self.head_q = builder.build_head(head)

        self.encoder_k = builder.build_backbone(backbone)    
        self.head_k = builder.build_head(head)

        if pretrained is not None:
            # TODO : check if loading works normally
            self.logger.info('load model from: {}'.format(pretrained))
            load_checkpoint(self.backbone, pretrained, strict=False, map_location='cpu', 
                            logger=self.logger, revise_keys=[(r'^backbone\.', '')])
        
        self.head_q.init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.head_q.parameters(),
                                    self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.regioncl_loss = builder.build_loss(regioncl_loss_cfg)

        self.queue_len = queue_len
        self.momentum = momentum
        self.cutMixUpper = cutMixUpper
        self.cutMixLower = cutMixLower

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.head_q.parameters(),
                                    self.head_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def EMNA_momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder's state_dict. In MoCo, it is parameters
        """
        for module_q, module_k in zip([self.encoder_q, self.head_q],
                                      [self.encoder_k, self.head_k]):
            state_dict_q = module_q.state_dict()
            state_dict_k = module_k.state_dict()
            for (k_q, v_q), (k_k, v_k) in zip(state_dict_q.items(), state_dict_k.items()):
                assert k_k == k_q, "state_dict names are different!"
                if 'num_batches_tracked' in k_k:
                    v_k.copy_(v_q)
                else:
                    v_k.copy_(v_k * self.momentum + (1. - self.momentum) * v_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size >= self.queue_len:
            self.queue = torch.cat((self.queue[:, ptr:], self.queue[:, :ptr]), dim=1)
            ptr = 0

        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def RegionSwapping(self, img):
        '''
        RegionSwapping(img)
        Args:
        :param img: [B, C, H, W]

        Return:
        :param img_mix: [B, C, H, W]
        '''

        B, C, H, W = img.shape
        randperm = torch.arange(B - 1, -1, -1)
        unshuffle = torch.argsort(randperm)
        randWidth = (self.downsampling * torch.randint(self.cutMixLower, self.cutMixUpper, (1,)).float())
        randHeight = (self.downsampling * torch.randint(self.cutMixLower, self.cutMixUpper, (1,)).float())

        randStartW = torch.randint(0, W, (1,)).float()
        randStartW = torch.round(randStartW / self.downsampling) * self.downsampling
        randStartW = torch.minimum(randStartW, W - 1 - randWidth)

        randStartH = torch.randint(0, H, (1,)).float()
        randStartH = torch.round(randStartH / self.downsampling) * self.downsampling
        randStartH = torch.minimum(randStartH, H - 1 - randHeight)

        randStartW = randStartW.long()
        randStartH = randStartH.long()
        randWidth = randWidth.long()
        randHeight = randHeight.long()

        img_mix = img.clone()
        img_mix[:, :, randStartH:randStartH + randHeight, randStartW:randStartW + randWidth] = \
            img[randperm, :, randStartH:randStartH + randHeight, randStartW:randStartW + randWidth]

        return img_mix, randStartW.float() / self.downsampling, randStartH.float() / self.downsampling, \
            randWidth.float() / self.downsampling, randHeight.float() / self.downsampling, randperm, unshuffle

    def forward_train(self, img, img_metas, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        losses = dict()
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        im_q_swapped, randStartW, randStartH, randWidth, randHeight, randperm, unShuffle = self.RegionSwapping(im_q)

        q = self.encoder_q(im_q)
        if self.input_transform:
            q = [self._transform_inputs(q)]
        else:
            q = [q[0]]

        q = self.head_q(q)

        q_swapped = self.encoder_q(im_q_swapped)
        if self.input_transform:
            q_swapped = [self._transform_inputs(q_swapped)]
        else:
            q_swapped = [q_swapped[0]]
            
        q_canvas, q_canvas_shuffle, q_paste, q_paste_shuffle = self.head_q(
            q_swapped, randStartW.long(), randStartH.long(), randWidth.long(), randHeight.long(), randperm, unShuffle)    # queries: NxC

        q = nn.functional.normalize(q, dim=1)
        q_canvas = nn.functional.normalize(q_canvas, dim=1)
        q_canvas_shuffle = nn.functional.normalize(q_canvas_shuffle, dim=1)
        q_paste = nn.functional.normalize(q_paste, dim=1)
        q_paste_shuffle = nn.functional.normalize(q_paste_shuffle, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
            if torch.cuda.device_count() > 1:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)
            if self.input_transform:
                k = [self._transform_inputs(k)]
            else:
                k = [k[0]]

            k = self.head_k(k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            if torch.cuda.device_count() > 1:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_instance = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_pos_region_canvas = torch.einsum('nc,nc->n', [q_canvas, k]).unsqueeze(-1)
        l_pos_region_paste = torch.einsum('nc,nc->n', [q_paste, k]).unsqueeze(-1)

        l_pos_region = torch.cat([l_pos_region_canvas, l_pos_region_paste], dim=0)

        # negative logits: NxK
        queue = self.queue.clone().detach()

        l_neg_instance = torch.einsum('nc,ck->nk', [q, queue])

        l_neg_canvas_inter = torch.einsum('nc,ck->nk', [q_canvas, queue])
        l_neg_canvas_intra = torch.einsum('nc,nc->n', [q_canvas, q_paste_shuffle.detach()]).unsqueeze(-1)
        l_neg_canvas = torch.cat([l_neg_canvas_intra, l_neg_canvas_inter], dim=1)

        l_neg_paste_inter = torch.einsum('nc,ck->nk', [q_paste, queue])
        l_neg_paste_intra = torch.einsum('nc,nc->n', [q_paste, q_canvas_shuffle.detach()]).unsqueeze(-1)
        l_neg_paste = torch.cat([l_neg_paste_intra, l_neg_paste_inter], dim=1)

        l_neg_region = torch.cat([l_neg_canvas, l_neg_paste], dim=0)

        losses['loss_contra_instance'] = self.regioncl_loss(l_pos_instance, l_neg_instance)['loss']
        losses['loss_contra_region'] = self.regioncl_loss(l_pos_region, l_neg_region)['loss']

        self._dequeue_and_enqueue(k)

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
class RegionCLMSupCon(RegionCLM):
    def forward_train(self, img, img_metas, gt_semantic_seg, pixel_weights=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        losses = dict()
        im_q, im_k = img[:, 0].contiguous(), img[:, 1].contiguous()
        mask_q = gt_semantic_seg[:, 0]

        im_q_swapped, randStartW, randStartH, randWidth, randHeight, randperm, unShuffle = self.RegionSwapping(im_q)

        q = self.encoder_q(im_q)

        # for supervised loss
        loss_decode, _ = self._decode_head_forward_train(
            q, img_metas, gt_semantic_seg=mask_q, pixel_weights=pixel_weights)
        
        losses.update(loss_decode)

        # for regioncl loss
        q = [self.decode_head._transform_inputs(q)]
        q = self.head_q(q)

        q_swapped = self.encoder_q(im_q_swapped)
        q_swapped = [self.decode_head._transform_inputs(q_swapped)]
        q_canvas, q_canvas_shuffle, q_paste, q_paste_shuffle = self.head_q(
            q_swapped, randStartW.long(), randStartH.long(), randWidth.long(), randHeight.long(), randperm, unShuffle)    # queries: NxC

        q = nn.functional.normalize(q, dim=1)
        q_canvas = nn.functional.normalize(q_canvas, dim=1)
        q_canvas_shuffle = nn.functional.normalize(q_canvas_shuffle, dim=1)
        q_paste = nn.functional.normalize(q_paste, dim=1)
        q_paste_shuffle = nn.functional.normalize(q_paste_shuffle, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
            if torch.cuda.device_count() > 1:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)
            k = [self.decode_head._transform_inputs(k)]

            k = self.head_k(k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            if torch.cuda.device_count() > 1:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_instance = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_pos_region_canvas = torch.einsum('nc,nc->n', [q_canvas, k]).unsqueeze(-1)
        l_pos_region_paste = torch.einsum('nc,nc->n', [q_paste, k]).unsqueeze(-1)

        l_pos_region = torch.cat([l_pos_region_canvas, l_pos_region_paste], dim=0)

        # negative logits: NxK
        queue = self.queue.clone().detach()

        l_neg_instance = torch.einsum('nc,ck->nk', [q, queue])

        l_neg_canvas_inter = torch.einsum('nc,ck->nk', [q_canvas, queue])
        l_neg_canvas_intra = torch.einsum('nc,nc->n', [q_canvas, q_paste_shuffle.detach()]).unsqueeze(-1)
        l_neg_canvas = torch.cat([l_neg_canvas_intra, l_neg_canvas_inter], dim=1)

        l_neg_paste_inter = torch.einsum('nc,ck->nk', [q_paste, queue])
        l_neg_paste_intra = torch.einsum('nc,nc->n', [q_paste, q_canvas_shuffle.detach()]).unsqueeze(-1)
        l_neg_paste = torch.cat([l_neg_paste_intra, l_neg_paste_inter], dim=1)

        l_neg_region = torch.cat([l_neg_canvas, l_neg_paste], dim=0)

        losses['loss_contra_instance'] = self.regioncl_loss(l_pos_instance, l_neg_instance)['loss']
        losses['loss_contra_region'] = self.regioncl_loss(l_pos_region, l_neg_region)['loss']

        self._dequeue_and_enqueue(k)

        return losses


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.cuda.device_count() > 1:
        tensors_gather = [
            torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor
        
    return output
