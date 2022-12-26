# Copyright (c) OpenMMLab. All rights reserved.
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Any, Union

import mmcv
import numpy as np

from .utils import is_mmdeploy_enabled, mmdeploy_init_model_helper


if is_mmdeploy_enabled():
    import mmdeploy.apis.openvino as openvino_api
    from mmdeploy.apis import build_task_processor, torch2onnx
    from mmdeploy.apis.openvino import get_input_info_from_cfg, get_mo_options_from_cfg
    from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.utils import get_ir_config

    def export2openvino(output_path, onnx_file_name, model_builder, cfg, deploy_cfg):

        task_processor = build_task_processor(cfg, deploy_cfg, "cpu")

        def helper(*args, **kwargs):
            return mmdeploy_init_model_helper(
                *args, **kwargs, model_builder=model_builder
            )

        task_processor.__class__.init_pytorch_model = helper

        input_data_cfg = deploy_cfg.pop(
            "input_data", {"shape": (128, 128, 3), "file_path": None}
        )
        if input_data_cfg.get("file_path"):
            import cv2

            input_data = cv2.imread(input_data_cfg.get("file_path"))
        else:
            input_data = np.zeros(input_data_cfg.shape, dtype=np.uint8)

        torch2onnx(
            input_data,
            output_path,
            onnx_file_name,
            deploy_cfg=deploy_cfg,
            model_cfg=cfg,
            model_checkpoint=cfg.load_from,
            device="cpu",
        )

        onnx2openvino(output_path, onnx_file_name, deploy_cfg)

    def onnx2openvino(
        work_dir: str,
        onnx_file: str,
        deploy_cfg: Union[str, mmcv.Config],
    ):

        onnx_path = os.path.join(work_dir, onnx_file)

        input_info = get_input_info_from_cfg(deploy_cfg)
        output_names = get_ir_config(deploy_cfg).output_names
        mo_options = get_mo_options_from_cfg(deploy_cfg)
        openvino_api.from_onnx(
            onnx_path, work_dir, input_info, output_names, mo_options
        )

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
