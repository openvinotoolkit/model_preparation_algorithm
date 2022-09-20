# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional
import math

import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmcls.models.builder import HEADS
from mmcls.models.heads import MultiLabelClsHead

class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class GroupASL(nn.Module):
    def __init__(self, embed_len_decoder: int):
        super().__init__()
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            h_i = F.normalize(h_i.view(h_i.shape[0], -1), dim=1)
            w_i = F.normalize(w_i, p=2., dim=0)
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class GroupFC(nn.Module):
    def __init__(self, embed_len_decoder: int):
        super().__init__()
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


@HEADS.register_module()
class CustomMultiLabelGroupFCClsHead(MultiLabelClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 hid_channels=1280,
                 num_of_groups=-1,
                 decoder_embedding=768,
                 normalized=False,
                 scale=1.0,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        super().__init__(loss=loss)
        self.num_classes = num_classes
        self.normalized = normalized
        self.scale = scale
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > self.num_classes:
            embed_len_decoder = self.num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(hid_channels, decoder_embedding)

        # non-learnable queries
        query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        query_embed.requires_grad_(False)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.llrelu = nn.LeakyReLU(0.2)

        # group fully-connected
        self.decoder.num_classes = self.num_classes
        self.decoder.duplicate_factor = int(self.num_classes / embed_len_decoder + 0.999)
        self.decoder.duplicate_pooling = torch.nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
        self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(self.num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        if self.normalized:
            self.decoder.group_fc = GroupASL(embed_len_decoder)
        else:
            self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def init_weights(self):
        nn.init.xavier_normal_(self.duplicate_pooling)
        nn.init.constant_(self.duplicate_pooling_bias, 0.)

    def loss(self, cls_score, gt_label, valid_label_mask=None):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, valid_label_mask=valid_label_mask, avg_factor=num_samples)
        losses['loss'] = loss / self.scale
        return losses

    def forward_train(self, x, gt_label, **kwargs):
        img_metas = kwargs.get('img_metas', False)
        gt_label = gt_label.type_as(x)

        spat_features = x
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        if self.normalized:
            embedding_spatial_786 = self.llrelu(embedding_spatial_786)
        else:
            embedding_spatial_786 = F.relu(embedding_spatial_786, inplace=True)
        bs = embedding_spatial_786.shape[0]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        cls_score = logits * self.scale

        if img_metas:
            valid_label_mask = self.get_valid_label_mask(img_metas=img_metas)
            losses = self.loss(cls_score, gt_label, valid_label_mask=valid_label_mask)
        else:
            losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, x):
        """Test without augmentation."""

        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        if self.normalized:
            embedding_spatial_786 = self.llrelu(embedding_spatial_786)
        else:
            embedding_spatial_786 = F.relu(embedding_spatial_786, inplace=True)
        bs = embedding_spatial_786.shape[0]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        cls_score = logits * self.scale
        pred = torch.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_valid_label_mask(self, img_metas):
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if 'ignored_labels' in meta and meta['ignored_labels']:
                mask[meta['ignored_labels']] = 0
            mask = mask.cuda() if torch.cuda.is_available() else mask
            valid_label_mask.append(mask)
        valid_label_mask = torch.stack(valid_label_mask, dim=0)
        return valid_label_mask
