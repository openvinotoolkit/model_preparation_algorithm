# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import warnings
import traceback
from functools import partial

import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmcv.runner import load_checkpoint
from mpa.registry import STAGES
from mpa.utils.logger import get_logger

from .stage import SegStage


logger = get_logger()


def build_segmentor(config, checkpoint=None, device="cpu", cfg_options=None):
    from mmseg.models import build_segmentor as origin_build_segmentor
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    model = origin_build_segmentor(config.model)
    model = model.to(device)
    if checkpoint is not None:
        load_checkpoint(model=model, filename=checkpoint, map_location=device)
    return model


@STAGES.register_module()
class SegExporter(SegStage):

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        self._init_logger()
        logger.info("exporting the model")
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"mode for this stage {mode}")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        output_path = os.path.join(cfg.work_dir, "export")
        os.makedirs(output_path, exist_ok=True)

        model_builder = kwargs.get("model_builder", build_segmentor)

        from torch.jit._trace import TracerWarning
        warnings.filterwarnings("ignore", category=TracerWarning)
        precision = kwargs.pop("precision", "FP32")
        if precision != "FP32":
            raise NotImplementedError
        logger.info(f"Model will be exported with precision {precision}")
        onnx_file_name = cfg.get("model_name", "model") + ".onnx"

        try:
            deploy_cfg = kwargs.get("deploy_cfg", None)
            if deploy_cfg is not None:
                self._mmdeploy_export(
                    output_path, onnx_file_name, model_builder, cfg, deploy_cfg
                )
            else:
                self._naive_export(output_path, onnx_file_name, model_builder, cfg)
        except Exception as ex:
            if (
                len([f for f in os.listdir(output_path) if f.endswith(".bin")]) == 0
                and len([f for f in os.listdir(output_path) if f.endswith(".xml")]) == 0
            ):
                # output_model.model_status = ModelStatus.FAILED
                # raise RuntimeError('Optimization was unsuccessful.') from ex
                return {
                    "outputs": None,
                    "msg": f"exception {type(ex)}: {ex}\n\n{traceback.format_exc()}"
                }

        bin_file = [f for f in os.listdir(output_path) if f.endswith(".bin")][0]
        xml_file = [f for f in os.listdir(output_path) if f.endswith(".xml")][0]
        logger.info("Exporting completed")
        return {
            "outputs": {
                "bin": os.path.join(output_path, bin_file),
                "xml": os.path.join(output_path, xml_file),
            },
            "msg": "",
        }

    def _mmdeploy_export(self, output_path, onnx_file_name, model_builder, cfg, deploy_cfg):
        from mpa.deploy.apis import onnx2openvino
        from mpa.deploy.utils import init_pytorch_model
        from mmdeploy.apis import build_task_processor, torch2onnx

        task_processor = build_task_processor(cfg, deploy_cfg, "cpu")
        task_processor.__class__.init_pytorch_model = partial(
            init_pytorch_model,
            model_builder=model_builder
        )

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

    def _naive_export(self, output_path, onnx_file_name, model_builder, cfg):
        raise NotImplementedError()
