# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
import torch

import mmcv
from mmcv.parallel import MMDataParallel, is_module_wrapper
from mmcv.runner import load_checkpoint

from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor, ImageTilingDataset
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
    'classwisesaliencymap': DetSaliencyMapHook,
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

        self.explain_dataset = build_dataset(cfg.data.test)
        explain_dataset = self.explain_dataset

        # Data loader
        explain_data_loader = build_dataloader(
            explain_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

        # Model
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        model = build_detector(cfg.model)
        DetectionStage.set_inference_progress_callback(model, cfg)

        # Checkpoint
        if cfg.get('load_from', None):
            load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        if torch.cuda.is_available():
            explain_model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        else:
            explain_model = MMDataCPU(model)

        if is_module_wrapper(model):
            model = model.module
        saliency_maps = []
        with self.explainer_hook(explain_model.module) as forward_explainer_hook:
            with torch.no_grad():
                _ = single_gpu_test(explain_model, explain_data_loader)
                saliency_maps = forward_explainer_hook.records

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(explain_dataset, 'dataset') and not isinstance(explain_dataset, ImageTilingDataset):
            explain_dataset = explain_dataset.dataset

        if isinstance(explain_dataset, ImageTilingDataset):
            saliency_maps = [saliency_maps[i] for i in range(explain_dataset.num_samples)]

        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs
