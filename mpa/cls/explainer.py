# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from os import path as osp

from mpa.registry import STAGES
from .inferrer import ClsInferrer

from mpa.utils.logger import get_logger
from contextlib import nullcontext

import os.path as osp
import numpy as np
import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

from mpa.registry import STAGES
from mpa.cls.stage import ClsStage
from mpa.modules.hooks.auxiliary_hooks import FeatureVectorHook, SaliencyMapHook
from mpa.utils.logger import get_logger
from mpa.modules.xai.builder import build_explainer

logger = get_logger()


@STAGES.register_module()
class ClsExplainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explainer stage
        """
        self._init_logger()
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._explain(cfg)
        return dict(
            outputs=outputs
            )

    def _explain(self, cfg):
        self.dataset = build_dataset(cfg.data.saliency_test)

        # Data loader
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False,
            persistent_workers=False)

        # build the model and load checkpoint
        model = build_classifier(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if cfg.load_from is not None:
            logger.info('load checkpoint from ' + cfg.load_from)
            _ = load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        model = MMDataParallel(model, device_ids=[0])
        explainer = build_explainer(model, cfg.explainer)
        explainer.eval()

        saliency_maps = []
        for data in data_loader:
            with torch.no_grad():
                out = [explainer(sample) for sample in data]
            saliency_maps.extend(out)

        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs
