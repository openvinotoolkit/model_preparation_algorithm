# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (C) 2022-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib


def is_mmdeploy_enabled():
    return importlib.util.find_spec("mmdeploy") is not None


def init_pytorch_model(ctx, model_checkpoint=None, cfg_options=None, **kwargs):
    model_builder = kwargs.pop("model_builder")
    model = model_builder(
        ctx.model_cfg,
        checkpoint=model_checkpoint,
        device=ctx.device,
        cfg_options=cfg_options,
    )

    # TODO: Need to investigate it why
    # NNCF compressed model lost trace context from time to time with no reason
    # even with 'torch.no_grad()'. Explicitly setting 'requires_grad' to'False'
    # makes things easier.
    for i in model.parameters():
        i.requires_grad = False

    return model
