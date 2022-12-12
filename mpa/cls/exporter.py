# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import warnings
import torch.onnx
import traceback
from functools import partial

from mmcv.runner import load_checkpoint, wrap_fp16_model

from mpa.registry import STAGES
from .stage import ClsStage
from mpa.utils import mo_wrapper
from mpa.utils.logger import get_logger
import numpy as np
import torch
from mmcls.datasets.pipelines import Compose
logger = get_logger()


def build_classifier(config):
    from mmcls.models import build_classifier as origin_build_classifier
    model = origin_build_classifier(config.model)
    load_checkpoint(model=model, filename=config.load_from, map_location="cpu")
    return model


@STAGES.register_module()
class ClsExporter(ClsStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_fake_input(self, cfg, orig_img_shape=(128, 128, 3)):
        pipeline = cfg.data.test.pipeline
        pipeline = Compose(pipeline)
        data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
        data = pipeline(data)
        return data

    def get_norm_values(self, cfg):
        pipeline = cfg.data.test.pipeline
        mean_values = [0, 0, 0]
        scale_values = [1, 1, 1]
        for pipeline_step in pipeline:
            if pipeline_step.type == 'Normalize':
                mean_values = pipeline_step.mean
                scale_values = pipeline_step.std
                break
        return mean_values, scale_values

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run exporter stage

        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            logger.warning(f'mode for this stage {mode}')
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        output_path = os.path.join(cfg.work_dir, 'export')
        os.makedirs(output_path, exist_ok=True)

        model_builder = kwargs.get("model_builder", None)
        if model_builder is None:
            model = build_classifier(cfg)
        else:
            model = model_builder(cfg)

        if hasattr(model, 'is_export'):
            model.is_export = True

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

        bin_file = [f for f in os.listdir(output_path) if f.endswith('.bin')][0]
        xml_file = [f for f in os.listdir(output_path) if f.endswith('.xml')][0]
        logger.info('Exporting completed')
        return {
            'outputs': {
                'bin': os.path.join(output_path, bin_file),
                'xml': os.path.join(output_path, xml_file),
            },
            'msg': ''
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

        model.forward = partial(model.forward, img_metas={}, return_loss=False)

        data = self.get_fake_input(cfg)
        fake_img = data['img'].unsqueeze(0)

        torch.onnx.export(model,
                          fake_img,
                          os.path.join(output_path, onnx_file_name),
                          verbose=False,
                          export_params=True,
                          input_names=['data'],
                          output_names=['logits', 'feature_vector', 'saliency_map'],
                          dynamic_axes={},
                          opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX
                          )

        mean_values, scale_values = self.get_norm_values(cfg)
        mo_args = {
            'input_model': os.path.join(output_path, onnx_file_name),
            'mean_values': mean_values,
            'scale_values': scale_values,
            #  'data_type': "FP32",
            'model_name': 'model',
            'reverse_input_channels': None,
        }

        ret, msg = mo_wrapper.generate_ir(output_path, output_path, silent=True, **mo_args)
