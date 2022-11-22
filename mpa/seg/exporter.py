# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import warnings
import traceback

import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mpa.registry import STAGES
from mpa.utils.logger import get_logger

from .stage import SegStage


logger = get_logger()


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
        if kwargs.get("model", None) is not None:
            model = kwargs.get("model")
        else:
            model = build_segmentor(cfg.model)
            if model_ckpt:
                logger.info(f"model checkpoint load: {model_ckpt}")
                load_checkpoint(model=model, filename=model_ckpt, map_location="cpu")

        from torch.jit._trace import TracerWarning
        warnings.filterwarnings("ignore", category=TracerWarning)
        model = model.cpu()
        model.eval()
        precision = kwargs.pop("precision", "FP32")
        if precision != "FP32":
            raise NotImplementedError
        logger.info(f"Model will be exported with precision {precision}")
        onnx_file_name = cfg.get("model_name", "model") + ".onnx"

        try:
            deploy_cfg = kwargs.get("deploy_cfg", None)
            if deploy_cfg is not None:
                self._mmdeploy_export(
                    output_path, onnx_file_name, model, cfg, deploy_cfg
                )
            else:
                self._naive_export(output_path, onnx_file_name, model, cfg)
        except Exception as ex:
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

    def _mmdeploy_export(self, output_path, onnx_file_name, model, cfg, deploy_cfg):
        from mpa.deploy.apis import onnx2openvino, torch2onnx

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
            model=model,
            device="cpu",
        )

        onnx2openvino(output_path, onnx_file_name, deploy_cfg)

    def _naive_export(self, output_path, onnx_file_name, model, cfg):
        raise NotImplementedError()
