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
from mpa.modules.utils.task_adapt import prob_extractor
from mpa.utils.logger import get_logger
logger = get_logger()


@STAGES.register_module()
class ClsInferrer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage

        - Configuration
        - Environment setup
        - Run inference via mmcls -> mmcv
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._infer(cfg)
        if cfg.get('task_adapt', False) and self.extract_prob:
            output_file_path = osp.join(cfg.work_dir, 'pre_stage_res.npy')
            np.save(output_file_path, outputs, allow_pickle=True)
            return dict(pre_stage_res=output_file_path)
        else:
            # output_file_path = osp.join(cfg.work_dir, 'infer_result.npy')
            # np.save(output_file_path, outputs, allow_pickle=True)
            return dict(
                # output_file_path=output_file_path,
                outputs=outputs
                )

    def _infer(self, cfg):
        if cfg.get('task_adapt', False) and not hasattr(self, 'eval'):
            dataset_cfg = cfg.data.train.copy()
            dataset_cfg.pipeline = cfg.data.test.pipeline
            self.dataset = build_dataset(dataset_cfg)
        else:
            self.dataset = build_dataset(cfg.data.test)

        # Data loader
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False)

        # build the model and load checkpoint
        model = build_classifier(cfg.model)
        self.extract_prob = hasattr(model, 'extract_prob')
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        logger.info('load checkpoint from ' + cfg.load_from)
        _ = load_checkpoint(model, cfg.load_from, map_location='cpu')

        model = MMDataParallel(model, device_ids=[0])
        if cfg.get('task_adapt', False) and not hasattr(self, 'eval') and self.extract_prob:
            old_prob, feats = prob_extractor(model.module, data_loader)
            data_infos = self.dataset.data_infos
            # pre-stage for LwF
            for i, data_info in enumerate(data_infos):
                data_info['soft_label'] = {task: value[i] for task, value in old_prob.items()}
            outputs = data_infos
        else:
            outputs = self.single_gpu_test(model, data_loader)

        return outputs

    def single_gpu_test(self, model, data_loader):
        model.eval()
        results = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            batch_size = len(result)
            if not isinstance(result, dict):
                result = np.array(result, dtype=np.float32)
                for r in result:
                    results.append(r)
            for _ in range(batch_size):
                prog_bar.update()

        if not isinstance(result, dict):
            result = np.array(result)
        return results
