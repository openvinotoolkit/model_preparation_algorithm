# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
import torch

import mmcv
from mmcv.parallel import MMDataParallel, is_module_wrapper
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU

from mpa.modules.hooks.recording_forward_hooks import ActivationMapHook, EigenCamHook
from mpa.registry import STAGES
from .stage import DetectionStage
from mpa.utils.logger import get_logger

logger = get_logger()
EXPLAINER_HOOK_SELECTOR = {
    'eigencam': EigenCamHook,
    'activationmap': ActivationMapHook,
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

    def explain(self, cfg):
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        data_cfg = cfg.data.test.copy()
        # Input source
        if 'input_source' in cfg:
            input_source = cfg.get('input_source')
            logger.info(f'Inferring on input source: data.{input_source}')
            src_data_cfg = cfg.data[input_source]
            data_cfg.test_mode = True
            data_cfg.ann_file = src_data_cfg.ann_file
            data_cfg.img_prefix = src_data_cfg.img_prefix
            if 'classes' in src_data_cfg:
                data_cfg.classes = src_data_cfg.classes
        self.explain_dataset = build_dataset(data_cfg)

        # Data loader
        explain_data_loader = build_dataloader(
            self.explain_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # Target classes
        if 'task_adapt' in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(f'target_classes={target_classes} is empty check the metadata from model ckpt or recipe '
                               f'configuration')
        else:
            target_classes = self.explain_dataset.CLASSES

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
        model.CLASSES = target_classes
        DetectionStage.set_inference_progress_callback(model, cfg)

        # Checkpoint
        if cfg.get('load_from', None):
            load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        if torch.cuda.is_available():
            model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        else:
            model = MMDataCPU(model)

        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        if is_module_wrapper(model):
            model = model.module

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
