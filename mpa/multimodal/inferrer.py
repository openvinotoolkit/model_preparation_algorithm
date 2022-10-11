import os.path as osp
import numpy as np
import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mpa.registry import STAGES
from mpa.multimodal.stage import MultimodalStage
from mpa.utils.logger import get_logger
from mpa.multimodal.trainer import check_modalities
from .builder import build_model, build_dataset, build_dataloader
logger = get_logger()


@STAGES.register_module()
class MultimodalInferrer(MultimodalStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}
        
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._infer(cfg)
        
        return dict(
            outputs=outputs
        )
    
    def _infer(self, cfg):
        cfg.data.test.modalities = check_modalities(cfg.data.test, phase='test')
        self.dataset = build_dataset(cfg.data.test)
        cfg.model.tabular_encoder.in_channels = self.dataset.table_df.shape[1]

        # Data loader
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )

        # build the model and load checkpoint
        cfg.model.modalities = cfg.data.test.modalities
        model = build_model(cfg.model)
        self.extract_prob = hasattr(model, 'extract_prob')
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if cfg.load_from is not None:
            logger.info('load checkpoint from ' + cfg.load_from)
            _ = load_checkpoint(model, cfg.load_from, map_location='cpu')

        model = MMDataParallel(model, device_ids=[0])

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        MultimodalInferrer.set_inference_progress_callback(model, cfg)
        
        outputs = self.single_gpu_test(model, data_loader)

        return outputs

    def single_gpu_test(self, model, data_loader):
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, **data)
            results.append(result)

            batch_size = data['meta_info'].size(0)
            for _ in range(batch_size):
                prog_bar.update()
        return results