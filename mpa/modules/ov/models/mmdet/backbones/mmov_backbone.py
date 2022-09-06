# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import BACKBONES

from ...mmov_model import MMOVModel


@BACKBONES.register_module()
class MMOVBackbone(MMOVModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_weights(self, pretrained=None):
        pass
