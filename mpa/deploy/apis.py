# Copyright (c) OpenMMLab. All rights reserved.
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Any, Union

import mmcv
import mmdeploy.apis.openvino as openvino_api
from mmdeploy.apis.openvino import (
    get_input_info_from_cfg,
    get_mo_options_from_cfg,
)
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import (
    get_ir_config,
)


def onnx2openvino(
    work_dir: str,
    onnx_file: str,
    deploy_cfg: Union[str, mmcv.Config],
):

    onnx_path = os.path.join(work_dir, onnx_file)

    input_info = get_input_info_from_cfg(deploy_cfg)
    output_names = get_ir_config(deploy_cfg).output_names
    mo_options = get_mo_options_from_cfg(deploy_cfg)
    openvino_api.from_onnx(onnx_path, work_dir, input_info, output_names, mo_options)


@FUNCTION_REWRITER.register_rewriter(
    "mmdeploy.core.optimizers.function_marker.mark_tensors", backend="openvino"
)
def remove_mark__openvino(ctx, xs: Any, *args, **kwargs):
    """Disable all marks for openvino backend

    As the Node `mark` is not able to be traced, we just return original input
    for the function `mark_tensors`.

    Args:
        xs (Any): Input structure which contains tensor.
    """
    return xs
