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
        """Run inference stage for detection

        - Configuration
        - Environment setup
        - Run inference via MMDetection -> MMCV
        """
        self._init_logger()
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        outputs = self.explain(cfg)

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
            if input_source == 'train':
                src_data_cfg = self.get_data_cfg(cfg, input_source)
            else:
                src_data_cfg = cfg.data[input_source]
            data_cfg.test_mode = src_data_cfg.get('test_mode', False)
            data_cfg.ann_file = src_data_cfg.ann_file
            data_cfg.img_prefix = src_data_cfg.img_prefix
            if 'classes' in src_data_cfg:
                data_cfg.classes = src_data_cfg.classes
        self.dataset = build_dataset(data_cfg)
        dataset = self.dataset

        # Data loader
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # Target classes
        target_classes = dataset.CLASSES

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

        # TODO: Check Inference FP16 Support
        # fp16_cfg = cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        DetectionStage.set_inference_progress_callback(model, cfg)

        # Checkpoint
        if cfg.get('load_from', None):
            load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        if torch.cuda.is_available():
            eval_model = MMDataParallel(model.cuda(cfg.gpu_ids[0]),
                                        device_ids=cfg.gpu_ids)
        else:
            eval_model = MMDataCPU(model)

        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        if is_module_wrapper(model):
            model = model.module

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        saliency_hook = DetSaliencyMapHook(eval_model.module)

        with saliency_hook:
            _ = single_gpu_test(eval_model, data_loader)
            saliency_maps = saliency_hook.records

        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
        ]:
            cfg.evaluation.pop(key, None)

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(dataset, 'dataset') and not isinstance(dataset, ImageTilingDataset):
            dataset = dataset.dataset

        if isinstance(dataset, ImageTilingDataset):
            saliency_maps = [saliency_maps[i] for i in range(dataset.num_samples)]

        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs
