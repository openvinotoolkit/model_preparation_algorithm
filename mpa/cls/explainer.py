# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmcls.datasets import build_dataloader, build_dataset

from mpa.registry import STAGES
from mpa.cls.stage import ClsStage, build_classifier
from mpa.modules.hooks.recording_forward_hooks import ActivationMapHook, EigenCamHook
from mpa.utils.logger import get_logger
logger = get_logger()
EXPLAINER_HOOK_SELECTOR = {
    'eigencam': EigenCamHook,
    'activationmap': ActivationMapHook,
}


@STAGES.register_module()
class ClsExplainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explain stage
        - Configuration
        - Environment setup
        - Run explain via hooks in recording_forward_hooks
        """
        self._init_logger()
        explainer = kwargs.get('explainer')
        self.explainer_hook = EXPLAINER_HOOK_SELECTOR.get(explainer.lower(), None)
        if self.explainer_hook is None:
            raise NotImplementedError(f'explainer algorithm {explainer} not supported')
        logger.info(
            f'explainer algorithm: {explainer}'
        )
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        model_builder = kwargs.get("model_builder", build_classifier)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._explain(cfg, model_builder)
        return dict(
            outputs=outputs
        )

    def _explain(self, cfg, model_builder):
        self.explain_dataset = build_dataset(cfg.data.test)

        # Data loader
        explain_data_loader = build_dataloader(
            self.explain_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False,
            persistent_workers=False)

        # build the model and load checkpoint
        model = model_builder(cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        self.extract_prob = hasattr(model, 'extract_prob')

        model.eval()
        model = MMDataParallel(model, device_ids=[0])

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        ClsStage.set_inference_progress_callback(model, cfg)
        with self.explainer_hook(model.module.backbone) as forward_explainer_hook:
            # do inference and record intermediate fmap
            for data in explain_data_loader:
                with torch.no_grad():
                    _ = model(return_loss=False, **data)
            saliency_maps = forward_explainer_hook.records

        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs
