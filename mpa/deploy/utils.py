# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (C) 2022-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

def is_mmdeploy_enabled():
    return importlib.util.find_spec("mmdeploy") is not None
