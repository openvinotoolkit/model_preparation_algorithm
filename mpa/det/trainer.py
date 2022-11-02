# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import numbers
import os
import os.path as osp
import re
import time
import datetime
from multiprocessing import Pipe, Process
import uuid

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env
from mpa.modules.utils.task_adapt import extract_anchor_ratio
from mpa.registry import STAGES
from mpa.stage import Stage
from mpa.utils.logger import get_logger

from .stage import DetectionStage

# TODO[JAEGUK]: Remove import detection_tasks
# from detection_tasks.apis.detection.config_utils import cluster_anchors


logger = get_logger()


@STAGES.register_module()
class DetectionTrainer(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for detection

        - Configuration
        - Environment setup
        - Run training via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        # Environment
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if world_size <= 1:
            distributed = False
        else:
            distributed = True
            gpu = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(gpu)
            dist.init_process_group(backend='nccl', world_size=world_size, rank=gpu)
            logger.info(f'dist info world_size = {dist.get_world_size()}, rank = {dist.get_rank()}')

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)
        logger.info('train!')

        # # Work directory
        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        logger.info(f'cfg.gpu_ids = {cfg.gpu_ids}, distributed = {distributed}')
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        if hasattr(cfg, 'hparams'):
            if cfg.hparams.get('adaptive_anchor', False):
                num_ratios = cfg.hparams.get('num_anchor_ratios', 5)
                proposal_ratio = extract_anchor_ratio(datasets[0], num_ratios)
                self.configure_anchor(cfg, proposal_ratio)

        # Dataset for HPO
        hp_config = kwargs.get('hp_config', None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)

        # Target classes
        if 'task_adapt' in cfg:
            target_classes = cfg.task_adapt.get('final', [])
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta['env_info'] = env_info
        # meta['config'] = cfg.pretty_text
        meta['seed'] = cfg.seed
        meta['exp_name'] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes)
            if 'proposal_ratio' in locals():
                cfg.checkpoint_config.meta.update({'anchor_ratio': proposal_ratio})

        if distributed:
            if cfg.dist_params.get('linear_scale_lr', False):
                new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
                logger.info(f'enabled linear scaling rule to the learning rate. \
                    changed LR from {cfg.optimizer.lr} to {new_lr}')
                cfg.optimizer.lr = new_lr

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.py'))
        # logger.info(f'Config:\n{cfg.pretty_text}')

        # run GPU utilization getter process
        parent_conn, child_conn = Pipe()
        p = Process(target=self.calculate_average_gpu_util, args=(cfg.work_dir, child_conn,))
        p.start()

        DetectionTrainer.train_worker(
            None if not distributed else int(os.environ['LOCAL_RANK']),
            target_classes,
            datasets,
            cfg,
            distributed,
            True,
            timestamp,
            meta)


        # kill GPU utilization getter process
        parent_conn.send(True)
        p.join()

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, 'latest.pth')
        best_ckpt_path = glob.glob(osp.join(cfg.work_dir, 'best_*.pth'))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(final_ckpt=output_ckpt_path)

    @staticmethod
    def train_worker(gpu, target_classes, datasets, cfg, distributed=False,
                     validate=False, timestamp=None, meta=None):
        # model
        model = build_detector(cfg.model)
        model.CLASSES = target_classes
        # Do clustering for SSD model
        # TODO[JAEGUK]: Temporal Disable cluster_anchors for SSD model
        # if hasattr(cfg.model, 'bbox_head') and hasattr(cfg.model.bbox_head, 'anchor_generator'):
        #     if getattr(cfg.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
        #         train_cfg = Stage.get_train_data_cfg(cfg)
        #         train_dataset = train_cfg.get('otx_dataset', None)
        #         cfg, model = cluster_anchors(cfg, train_dataset, model)
        start_time = datetime.datetime.now()
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=True,
            timestamp=timestamp,
            meta=meta)
        with open(osp.join(cfg.work_dir, f"time_{uuid.uuid4().hex}.txt"), "wt") as f:
            f.write(str(datetime.datetime.now() - start_time))

    @staticmethod
    def calculate_average_gpu_util(work_dir: str, pipe: Pipe):
        pynvml.nvmlInit()
        logger.info("GPU utilzer process start")
        filter = re.compile(r"([-+]?\d+) \%.*([-+]?\d+) \%")

        available_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if available_gpu is None:
            gpu_devices = [i for i in range(pynvml.nvmlDeviceGetCount())]
        else:
            gpu_devices = [int(gpuidx) for gpuidx in available_gpu.split(',')]

        per_gpu_util = {gpuidx : 0 for gpuidx in gpu_devices}
        num_total_util = 0
        while True:
            for gpuidx in gpu_devices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpuidx)
                use = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result = filter.search(str(use))
                if result is not None:
                    per_gpu_util[gpuidx] += float(result.group(1))

            if pipe.poll():
                try:
                    report = pipe.recv()
                    if report:
                        break
                except EOFError:
                    break

            num_total_util += 1

            time.sleep(1)

        pynvml.nvmlShutdown()

        with open(osp.join(work_dir, f"gpu_util_{os.getpid()}.txt"), "wt") as f:
            for gpuidx, gpu_util in per_gpu_util.items():
                f.write(f"#{gpuidx} average GPU util : {gpu_util / num_total_util}\n")
            f.write(f"total average GPU util : {sum(per_gpu_util.values()) / (len(per_gpu_util) * num_total_util)}\n")

        logger.info("GPU utilzer process is done")
