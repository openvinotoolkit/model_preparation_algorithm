# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from mmcv import ConfigDict
from mmdet.datasets import build_dataset
from mpa.stage import Stage
from mpa.utils.config_utils import update_or_add_custom_hook
from mpa.utils.logger import get_logger

logger = get_logger()


class DetectionStage(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        logger.info(f'configure!: training={training}')

        # Recipe + model
        cfg = self.cfg
        if model_cfg:
            if hasattr(model_cfg, 'model'):
                cfg.merge_from_dict(model_cfg._cfg_dict)
            else:
                raise ValueError("Unexpected config was passed through 'model_cfg'. "
                                 "it should have 'model' attribute in the config")
            model_task = cfg.model.pop('task', 'detection')
            if model_task != 'detection':
                raise ValueError(
                    f'Given model_cfg ({model_cfg.filename}) is not supported by detection recipe'
                )
        self.configure_model(cfg, training, **kwargs)

        # Checkpoint
        if model_ckpt:
            cfg.load_from = self.get_model_ckpt(model_ckpt)
        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained  # Overriding by stage input

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        self.configure_data(cfg, training, **kwargs)

        # Task
        if 'task_adapt' in cfg:
            self.configure_task(cfg, training, **kwargs)

        # Regularization
        if training:
            self.configure_regularization(cfg)

        # Other hyper-parameters
        if 'hyperparams' in cfg:
            self.configure_hyperparams(cfg, training, **kwargs)

        # Hooks
        self.configure_hook(cfg)

        return cfg

    def configure_model(self, cfg, training, **kwargs):
        super_type = cfg.model.pop('super_type', None)
        if super_type:
            cfg.model.arch_type = cfg.model.type
            cfg.model.type = super_type

        # OMZ-plugin
        if cfg.model.backbone.type == 'OmzBackboneDet':
            ir_path = kwargs.get('ir_path')
            if not ir_path:
                raise RuntimeError('OMZ model needs OpenVINO bin/xml files.')
            cfg.model.backbone.model_path = ir_path
            if cfg.model.type == 'SingleStageDetector':
                cfg.model.bbox_head.model_path = ir_path
            elif cfg.model.type == 'FasterRCNN':
                cfg.model.rpn_head.model_path = ir_path
            else:
                raise NotImplementedError(f'Unknown model type - {cfg.model.type}')

    def configure_anchor(self, cfg, proposal_ratio=None):
        if cfg.model.type in ['SingleStageDetector', 'CustomSingleStageDetector']:
            anchor_cfg = cfg.model.bbox_head.anchor_generator
            if anchor_cfg.type == 'SSDAnchorGeneratorClustered':
                cfg.model.bbox_head.anchor_generator.pop('input_size', None)

    def configure_data(self, cfg, training, **kwargs):
        Stage.configure_data(cfg, training, **kwargs)
        super_type = cfg.data.train.pop('super_type', None)
        if super_type:
            cfg.data.train.org_type = cfg.data.train.type
            cfg.data.train.type = super_type
        if training:
            if 'unlabeled' in cfg.data and cfg.data.unlabeled.get('img_file', None):
                cfg.data.unlabeled.ann_file = cfg.data.unlabeled.pop('img_file')
                if len(cfg.data.unlabeled.get('pipeline', [])) == 0:
                    cfg.data.unlabeled.pipeline = cfg.data.train.pipeline.copy()
                update_or_add_custom_hook(
                    cfg,
                    ConfigDict(
                        type='UnlabeledDataHook',
                        unlabeled_data_cfg=cfg.data.unlabeled,
                        samples_per_gpu=cfg.data.unlabeled.pop('samples_per_gpu', cfg.data.samples_per_gpu),
                        workers_per_gpu=cfg.data.unlabeled.pop('workers_per_gpu', cfg.data.workers_per_gpu),
                        seed=cfg.seed
                    )
                )

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation
        """
        logger.info(f'task config!!!!: training={training}')
        task_adapt_type = cfg['task_adapt'].get('type', None)
        task_adapt_op = cfg['task_adapt'].get('op', 'REPLACE')

        # Task classes
        org_model_classes, model_classes, data_classes = \
            self.configure_task_classes(cfg, task_adapt_type, task_adapt_op)

        # Data pipeline
        if data_classes != model_classes:
            self.configure_task_data_pipeline(cfg, model_classes, data_classes)

        # Evaluation dataset
        if cfg.get('task', 'detection') == 'detection':
            self.configure_task_eval_dataset(cfg, model_classes)

        # Training hook for task adaptation
        self.configure_task_adapt_hook(cfg, org_model_classes, model_classes)

        # Anchor setting
        if cfg['task_adapt'].get('use_mpa_anchor', False):
            self.configure_anchor(cfg)

        # Incremental learning
        self.configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes)

    def configure_task_classes(self, cfg, task_adapt_type, task_adapt_op):

        # Input classes
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        # Model classes
        if task_adapt_op == 'REPLACE':
            if len(data_classes) == 0:
                model_classes = org_model_classes.copy()
            else:
                model_classes = data_classes.copy()
        elif task_adapt_op == 'MERGE':
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f'{task_adapt_op} is not supported for task_adapt options!')

        if task_adapt_type == 'mpa':
            data_classes = model_classes
        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        # Model architecture
        head_names = ('mask_head', 'bbox_head', 'segm_head')
        num_classes = len(model_classes)
        if 'roi_head' in cfg.model:
            # For Faster-RCNNs
            for head_name in head_names:
                if head_name in cfg.model.roi_head:
                    if isinstance(cfg.model.roi_head[head_name], list):
                        for head in cfg.model.roi_head[head_name]:
                            head.num_classes = num_classes
                    else:
                        cfg.model.roi_head[head_name].num_classes = num_classes
        else:
            # For other architectures (including SSD)
            for head_name in head_names:
                if head_name in cfg.model:
                    cfg.model[head_name].num_classes = num_classes

        return org_model_classes, model_classes, data_classes

    def configure_task_data_pipeline(self, cfg, model_classes, data_classes):
        # Trying to alter class indices of training data according to model class order
        tr_data_cfg = self.get_train_data_cfg(cfg)
        class_adapt_cfg = dict(type='AdaptClassLabels', src_classes=data_classes, dst_classes=model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, op in enumerate(pipeline_cfg):
            if op['type'] == 'LoadAnnotations':  # insert just after this op
                op_next_ann = pipeline_cfg[i + 1] if i + 1 < len(pipeline_cfg) else {}
                if op_next_ann.get('type', '') == class_adapt_cfg['type']:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)
                break

    def configure_task_eval_dataset(self, cfg, model_classes):
        # - Altering model outputs according to dataset class order
        eval_types = ['val', 'test']
        for eval_type in eval_types:
            if cfg.data[eval_type]['type'] == 'TaskAdaptEvalDataset':
                cfg.data[eval_type]['model_classes'] = model_classes
            else:
                # Wrap original dataset config
                org_type = cfg.data[eval_type]['type']
                cfg.data[eval_type]['type'] = 'TaskAdaptEvalDataset'
                cfg.data[eval_type]['org_type'] = org_type
                cfg.data[eval_type]['model_classes'] = model_classes

    def configure_task_adapt_hook(self, cfg, org_model_classes, model_classes):
        task_adapt_hook = ConfigDict(
            type='TaskAdaptHook',
            src_classes=org_model_classes,
            dst_classes=model_classes,
            model_type=cfg.model.type,
        )
        update_or_add_custom_hook(cfg, task_adapt_hook)

    def configure_task_cls_incr(self, cfg, task_adapt_type, org_model_classes, model_classes):
        if cfg.get('task', 'detection') == 'detection':
            bbox_head = cfg.model.bbox_head
        else:
            bbox_head = cfg.model.roi_head.bbox_head
        if task_adapt_type == 'mpa':
            tr_data_cfg = self.get_train_data_cfg(cfg)
            if tr_data_cfg.type != 'MPADetDataset':
                tr_data_cfg.img_ids_dict = self.get_img_ids_for_incr(cfg, org_model_classes, model_classes)
                tr_data_cfg.org_type = tr_data_cfg.type
                tr_data_cfg.type = 'DetIncrCocoDataset'
            alpha, gamma = 0.25, 2.0
            if bbox_head.type in ['SSDHead', 'CustomSSDHead']:
                gamma = 1 if cfg['task_adapt'].get('efficient_mode', False) else 2
                bbox_head.type = 'CustomSSDHead'
                bbox_head.loss_cls = ConfigDict(
                    type='FocalLoss',
                    loss_weight=1.0,
                    gamma=gamma,
                    reduction='none',
                )
            elif bbox_head.type in ['ATSSHead']:
                gamma = 3 if cfg['task_adapt'].get('efficient_mode', False) else 4.5
                bbox_head.loss_cls.gamma = gamma
            elif bbox_head.type in ['VFNetHead', 'CustomVFNetHead']:
                alpha = 0.75
                gamma = 1 if cfg['task_adapt'].get('efficient_mode', False) else 2

            # Ignore Mode
            if cfg.get('ignore', False):
                bbox_head.loss_cls = ConfigDict(
                        type='CrossSigmoidFocalLoss',
                        use_sigmoid=True,
                        num_classes=len(model_classes),
                        alpha=alpha,
                        gamma=gamma
                )
            update_or_add_custom_hook(
                cfg,
                ConfigDict(
                    type='TaskAdaptHook',
                    sampler_flag=True,
                    efficient_mode=cfg['task_adapt'].get('efficient_mode', False)
                )
            )
            update_or_add_custom_hook(cfg, ConfigDict(type='EMAHook'))
        else:
            src_data_cfg = Stage.get_train_data_cfg(cfg)
            src_data_cfg.pop('old_new_indices', None)

    def configure_regularization(self, cfg):
        if cfg.model.get('l2sp_weight', 0.0) > 0.0:
            logger.info('regularization config!!!!')

            # Checkpoint
            l2sp_ckpt = cfg.model.get('l2sp_ckpt', None)
            if l2sp_ckpt is None:
                if 'pretrained' in cfg.model:
                    l2sp_ckpt = cfg.model.pretrained
                if cfg.load_from:
                    l2sp_ckpt = cfg.load_from
            cfg.model.l2sp_ckpt = l2sp_ckpt

            # Disable weight decay
            if 'weight_decay' in cfg.optimizer:
                cfg.optimizer.weight_decay = 0.0

    @staticmethod
    def get_img_ids_for_incr(cfg, org_model_classes, model_classes):
        # get image ids of old classes & new class
        # to setup experimental dataset (COCO format)
        new_classes = np.setdiff1d(model_classes, org_model_classes).tolist()
        old_classes = np.intersect1d(org_model_classes, model_classes).tolist()

        src_data_cfg = Stage.get_train_data_cfg(cfg)

        ids_old, ids_new = [], []
        data_cfg = cfg.data.test.copy()
        data_cfg.test_mode = src_data_cfg.get('test_mode', False)
        data_cfg.ann_file = src_data_cfg.get('ann_file', None)
        data_cfg.img_prefix = src_data_cfg.get('img_prefix', None)
        old_data_cfg = data_cfg.copy()
        if 'classes' in old_data_cfg:
            old_data_cfg.classes = old_classes
        old_dataset = build_dataset(old_data_cfg)
        ids_old = old_dataset.dataset.img_ids
        if len(new_classes) > 0:
            data_cfg.classes = new_classes
            dataset = build_dataset(data_cfg)
            ids_new = dataset.dataset.img_ids
            ids_old = np.setdiff1d(ids_old, ids_new).tolist()

        sampled_ids = ids_old + ids_new
        outputs = dict(
            old_classes=old_classes,
            new_classes=new_classes,
            img_ids=sampled_ids,
            img_ids_old=ids_old,
            img_ids_new=ids_new,
        )
        return outputs

    def configure_hyperparams(self, cfg, training, **kwargs):
        hyperparams = kwargs.get('hyperparams', None)
        if hyperparams is not None:
            bs = hyperparams.get('bs', None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get('lr', None)
            if lr is not None:
                cfg.optimizer.lr = lr
