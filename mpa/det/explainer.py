# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU

from mpa.det.stage import DetectionStage
from mpa.modules.hooks.recording_forward_hooks import ActivationMapHook, EigenCamHook, DetSaliencyMapHook
from mpa.registry import STAGES
from mpa.utils.logger import get_logger
logger = get_logger()
EXPLAINER_HOOK_SELECTOR = {
    'eigencam': EigenCamHook,
    'activationmap': ActivationMapHook,
    'detsaliencymap': DetSaliencyMapHook,
}


@STAGES.register_module()
class DetectionExplainer(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explain stage for detection
        - Configuration
        - Environment setup
        - Run explain via MMDetection -> MMCV
        """
        self._init_logger()
        explainer = kwargs.get('explainer')
        self.explainer_hook = EXPLAINER_HOOK_SELECTOR.get(explainer.lower(), None)
        if self.explainer_hook is None:
            raise NotImplementedError(f'Explainer algorithm {explainer} not supported!')
        logger.info(
            f'Explainer algorithm: {explainer}'
        )
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._explain(cfg)
        return dict(
            outputs=outputs
        )

    def _explain(self, cfg):
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        data_cfg = cfg.data.test.copy()
        self.explain_dataset = build_dataset(data_cfg)
        explain_dataset = self.explain_dataset

        # Data loader
        explain_data_loader = build_dataloader(
            explain_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # Model
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            for neck_cfg in list(cfg.model.neck):
                if neck_cfg.get('rfp_backbone') and neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None

        model = build_detector(cfg.model)
        DetectionStage.set_inference_progress_callback(model, cfg)

        # Checkpoint
        if cfg.get('load_from', None):
            load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        if torch.cuda.is_available():
            explain_backbone = MMDataParallel(
                model.backbone.cuda(cfg.gpu_ids[0]),
                device_ids=cfg.gpu_ids
            )
        else:
            explain_backbone = MMDataCPU(model.backbone)

        saliency_maps = []
        with torch.no_grad():
            for data in explain_data_loader:
                out = explain_backbone(data['img'][0])
                saliency_maps.append(
                    self.explainer_hook.func(out[-1]).squeeze(0).detach().cpu().numpy()
                )
        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs