# Copyright (c) OpenMMLab. All rights reserved.
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Any, Optional, Union

import mmcv
import mmdeploy.apis.openvino as openvino_api
import torch
from mmdeploy.apis.core.pipeline_manager import no_mp
from mmdeploy.apis.onnx import export
from mmdeploy.apis.openvino import (
    get_input_info_from_cfg,
    get_mo_options_from_cfg,
    get_output_model_file,
)
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import (
    Backend,
    get_backend,
    get_dynamic_axes,
    get_input_shape,
    get_ir_config,
    load_config,
)


# mmdeploy.apis.pytorch2onnx
def torch2onnx(
    img: Any,
    work_dir: str,
    save_file: str,
    deploy_cfg: Union[str, mmcv.Config],
    model_cfg: Optional[Union[str, mmcv.Config]] = None,
    model_checkpoint: Optional[str] = None,
    model: Optional[Any] = None,
    device: str = "cuda:0",
):
    """Convert PyTorch model to ONNX model.

    Examples:
        >>> from mmdeploy.apis import torch2onnx
        >>> img = 'demo.jpg'
        >>> work_dir = 'work_dir'
        >>> save_file = 'fcos.onnx'
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_onnxruntime_dynamic.py')
        >>> model_cfg = ('mmdetection/configs/fcos/'
                         'fcos_r50_caffe_fpn_gn-head_1x_coco.py')
        >>> model_checkpoint = ('checkpoints/'
                                'fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth')
        >>> device = 'cpu'
        >>> torch2onnx(img, work_dir, save_file, deploy_cfg, \
            model_cfg, model_checkpoint, device)

    Args:
        img (str | np.ndarray | torch.Tensor): Input image used to assist
            converting model.
        work_dir (str): A working directory to save files.
        save_file (str): Filename to save onnx model.
        deploy_cfg (str | mmcv.Config): Deployment config file or
            Config object.
        model_cfg (str | mmcv.Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        device (str): A string specifying device type, defaults to 'cuda:0'.
    """
    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmcv.mkdir_or_exist(os.path.abspath(work_dir))

    input_shape = get_input_shape(deploy_cfg)

    # create model an inputs
    from mmdeploy.apis import build_task_processor

    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    if model is None:
        torch_model = task_processor.init_pytorch_model(model_checkpoint)
    else:
        torch_model = model

    data, model_inputs = task_processor.create_input(img, input_shape)
    if "img_metas" in data:
        input_metas = dict(img_metas=data["img_metas"])
    else:
        # codebases like mmedit do not have img_metas argument
        input_metas = None
    # FIXME: mmdet only
    if not isinstance(model_inputs, torch.Tensor) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]

    # export to onnx
    context_info = dict()
    context_info["deploy_cfg"] = deploy_cfg
    output_prefix = os.path.join(
        work_dir, os.path.splitext(os.path.basename(save_file))[0]
    )
    backend = get_backend(deploy_cfg).value

    onnx_cfg = get_ir_config(deploy_cfg)
    opset_version = onnx_cfg.get("opset_version", 11)

    input_names = onnx_cfg["input_names"]
    output_names = onnx_cfg["output_names"]
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get("strip_doc_string", True) or onnx_cfg.get(
        "verbose", False
    )
    keep_initializers_as_inputs = onnx_cfg.get("keep_initializers_as_inputs", True)
    optimize = onnx_cfg.get("optimize", False)
    if backend == Backend.NCNN.value:
        """NCNN backend needs a precise blob counts, while using onnx optimizer
        will merge duplicate initilizers without reference count."""
        optimize = False

    # TODO: Need to investigate it why
    # NNCF compressed model lost trace context from time to time with no reason
    # even with 'torch.no_grad()'. Explicitly setting 'requires_grad' to'False'
    # makes things easier.
    for i in torch_model.parameters():
        i.requires_grad = False

    with no_mp():
        export(
            torch_model,
            model_inputs,
            input_metas=input_metas,
            output_path_prefix=output_prefix,
            backend=backend,
            input_names=input_names,
            output_names=output_names,
            context_info=context_info,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            optimize=optimize,
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
