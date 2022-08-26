# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2021 Alibaba-MIIL
# SPDX-License-Identifier: MIT
#


import torch
import torch.nn as nn

from mmcls.models.builder import NECKS


@NECKS.register_module()
class MLDecoder(nn.Module):
    """ML-Decoder neck.
    See details in https://arxiv.org/pdf/2111.12933.pdf
    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, num_classes, in_channels, num_of_groups=100, decoder_embedding=768,
                 initial_num_features=2048):
        super().__init__()

        assert num_of_groups > 0
        assert decoder_embedding > 0

        embed_len_decoder = min(num_classes, num_of_groups)
        embed_standart = nn.Linear(in_channels, decoder_embedding)

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

    def init_weights(self):
        pass

    def forward(self, x):
        assert len(x.shape) == 4  # [bs,nc,h,w]
        embedding_spatial = x.flatten(2).transpose(1, 2)
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = F.relu(embedding_spatial_786, inplace=True)
        bs = embedding_spatial_786.shape[0]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)
        return h


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
            state['activation'] = F.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None):
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
