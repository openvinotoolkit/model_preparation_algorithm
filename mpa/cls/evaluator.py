# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from os import path as osp
import mmcv

from mpa.registry import STAGES
from .inferrer import ClsInferrer

from mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class ClsEvaluator(ClsInferrer):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage

        - Run inference
        - Get saliency map
        """
        self.eval = True
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            logger.warning(f'mode for this stage {mode}')
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Save config
        cfg.dump(osp.join(cfg.work_dir, 'config.yaml'))
        logger.info(f'Config:\n{cfg.pretty_text}')

        # Inference
        infer_results = super()._infer(cfg)
        
        # get saliency maps
        saliency_maps = infer_results['saliency_maps']
        self.dump_saliency_map(osp.join(cfg.work_dir, 'saliency_maps'), saliency_maps)
        
    @staticmethod
    def dump_saliency_map(root, saliency_maps):
        mmcv.mkdir_or_exist(root)
        from PIL import Image
        for i, saliency_map in enumerate(saliency_maps):
            img = Image.fromarray(saliency_map)
            img = img.resize((224, 224))
            img.save(osp.join(root, f'saliency_map{i}.png'))