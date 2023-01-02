# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import traceback
from copy import deepcopy

import numpy as np
from mmcv.runner import wrap_fp16_model
from mpa.registry import STAGES
from mpa.utils.logger import get_logger

from .stage import DetectionStage, build_detector


logger = get_logger()


@STAGES.register_module()
class DetectionExporter(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        self._init_logger()
        logger.info("exporting the model")
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"mode for this stage {mode}")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        output_dir = os.path.join(cfg.work_dir, "export")
        os.makedirs(output_dir, exist_ok=True)

        #  from torch.jit._trace import TracerWarning
        #  import warnings
        #  warnings.filterwarnings("ignore", category=TracerWarning)
        precision = kwargs.pop("precision", "FP32")
        if precision not in ("FP32", "FP16", "INT8"):
            raise NotImplementedError
        logger.info(f"Model will be exported with precision {precision}")
        model_name = cfg.get("model_name", "model")

        _model_builder = kwargs.get("model_builder", build_detector)

        def model_builder(*args, **kwargs):
            model = _model_builder(*args, **kwargs)

            if precision == "FP16":
                wrap_fp16_model(model)
            elif precision == "INT8":
                from nncf.torch.nncf_network import NNCFNetwork

                assert isinstance(model, NNCFNetwork)

            return model

        try:
            deploy_cfg = kwargs.get("deploy_cfg", None)
            if deploy_cfg is not None:
                self._mmdeploy_export(
                    output_dir,
                    model_builder,
                    precision,
                    cfg,
                    deploy_cfg,
                    model_name,
                )
            else:
                self._naive_export(
                    output_dir, model_builder, precision, cfg, model_name
                )
        except Exception as ex:
            if (
                len([f for f in os.listdir(output_dir) if f.endswith(".bin")]) == 0
                and len([f for f in os.listdir(output_dir) if f.endswith(".xml")]) == 0
            ):
                # output_model.model_status = ModelStatus.FAILED
                # raise RuntimeError('Optimization was unsuccessful.') from ex
                return {
                    "outputs": None,
                    "msg": f"exception {type(ex)}: {ex}\n\n{traceback.format_exc()}",
                }

        bin_file = [f for f in os.listdir(output_dir) if f.endswith(".bin")][0]
        xml_file = [f for f in os.listdir(output_dir) if f.endswith(".xml")][0]
        logger.info("Exporting completed")
        return {
            "outputs": {
                "bin": os.path.join(output_dir, bin_file),
                "xml": os.path.join(output_dir, xml_file),
            },
            "msg": "",
        }

    def _mmdeploy_export(
        self, output_dir, model_builder, precision, cfg, deploy_cfg, model_name,
    ):
        from mpa.deploy.apis import MMdeployExporter

        if precision == "FP16":
            deploy_cfg.backend_config.mo_options.flags.append("--compress_to_fp16")
        MMdeployExporter.export2openvino(
            output_dir, model_builder, cfg, deploy_cfg, model_name=model_name
        )

    def _naive_export(self, output_dir, model_builder, precision, cfg, model_name):
        from mmdet.datasets.pipelines import Compose
        from mmdet.apis.inference import LoadImage

        from ..deploy.apis import NaiveExporter
        from ..deploy.utils.mmdet_symbolic import (
            register_extra_symbolics_for_openvino,
            unregister_extra_symbolics_for_openvino
        )

        def get_fake_data(cfg, orig_img_shape=(128, 128, 3)):
            pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
            pipeline = Compose(pipeline)
            data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
            data = pipeline(data)
            return data

        fake_data = get_fake_data(cfg)
        opset_version = 11
        register_extra_symbolics_for_openvino(opset_version)

        NaiveExporter.export2openvino(
            output_dir,
            model_builder,
            cfg,
            fake_data,
            precision=precision,
            model_name=model_name,
            input_names=["image"],
            output_names=["boxes", "labels"],
            opset_version=opset_version,
        )

        unregister_extra_symbolics_for_openvino(opset_version)
