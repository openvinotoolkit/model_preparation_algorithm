import os
import time
import torch
import numpy as np
import logging
from mmcv import ConfigDict
from mmdet.utils import get_root_logger
from mmdet.datasets import build_dataset

from mpa.stage import Stage


class DetectionStage(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = None

    def configure(self, model_cfg, model_ckpt, data_cfg, training=True, **kwargs):
        """Create MMCV-consumable config from given inputs
        """
        self.logger.info(f'configure!: training={training}')

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
        configure_model(cfg, training, **kwargs)

        # Checkpoint
        if model_ckpt:
            cfg.load_from = model_ckpt
        pretrained = kwargs.get('pretrained', None)
        if pretrained:
            self.logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained  # Overriding by stage input

        # Data
        if data_cfg:
            cfg.merge_from_dict(data_cfg)
        configure_data(cfg, training, **kwargs)

        # Task
        if 'task_adapt' in cfg:
            configure_task(cfg, training, **kwargs)

        # Regularization
        if training:
            configure_regularization(cfg)

        # Other hyper-parameters
        if 'hyperparams' in cfg:
            configure_hyperparams(cfg, training, **kwargs)

        return cfg

    def _init_logger(self):
        ''' override to initalize mmdet logger instead of mpa one.
        '''
        if self.logger is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            work_dir = os.path.dirname(self.cfg.work_dir)
            # print(f'workdir for the logger of detection tasks: {work_dir}')
            log_lvl = logging.getLevelName(self.cfg.log_level.upper())
            self.logger = get_root_logger(log_file=os.path.join(work_dir, f'{timestamp}.log'),
                                          log_level=log_lvl)


def configure_model(cfg, training, **kwargs):
    super_type = cfg.model.pop('super_type', None)
    if super_type:
        cfg.model.arch_type = cfg.model.type
        cfg.model.type = super_type

    if not training:
        # BBox head for pseudo label output
        if 'roi_head' in cfg.model:
            # For Faster-RCNNs
            bbox_head_cfg = cfg.model.roi_head.bbox_head
        else:
            # For other architectures
            bbox_head_cfg = cfg.model.bbox_head

        if bbox_head_cfg.type in ['Shared2FCBBoxHead', 'PseudoShared2FCBBoxHead']:
            bbox_head_cfg.type = 'PseudoShared2FCBBoxHead'
        elif bbox_head_cfg.type in ['SSDHead', 'PseudoSSDHead']:
            bbox_head_cfg.type = 'PseudoSSDHead'

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


def configure_anchor(cfg, proposal_ratio=None):
    if cfg.model.type in ['SingleStageDetector', 'CustomSingleStageDetector']:
        anchor_cfg = cfg.model.bbox_head.anchor_generator
        if anchor_cfg.type == 'SSDAnchorGeneratorClustered':
            cfg.model.bbox_head.anchor_generator.pop('input_size', None)


def get_model_classes(cfg):
    """Extract trained classes info from checkpoint file.

    MMCV-based models would save class info in ckpt['meta']['CLASSES']
    For other cases, try to get the info from cfg.model.classes (with pop())
    - Which means that model classes should be specified in model-cfg for
      non-MMCV models (e.g. OMZ models)
    """
    classes = []
    ckpt_path = cfg.get('load_from', None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        meta = ckpt.get('meta', {})
        classes = meta.get('CLASSES', [])
    if len(classes) == 0:
        classes = cfg.model.pop('classes', [])
    return classes


def get_data_classes(cfg):
    # TODO: getting from actual dataset
    return list(get_train_data_cfg(cfg).get('classes', []))


def get_train_data_cfg(cfg):
    if 'dataset' in cfg.data.train:  # Concat|RepeatDataset
        return cfg.data.train.dataset
    else:
        return cfg.data.train


def configure_data(cfg, training, **kwargs):
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


def configure_task(cfg, training, **kwargs):
    """Adjust settings for task adaptation
    """
    logger = get_root_logger()
    logger.info(f'task config!!!!: training={training}')
    # target_classes = []
    task_adapt_type = cfg['task_adapt'].get('type', None)
    adapt_type = cfg['task_adapt'].get('op', 'REPLACE')
    org_model_classes = get_model_classes(cfg)
    data_classes = get_data_classes(cfg)

    # Model classes
    if adapt_type == 'REPLACE':
        if len(data_classes) == 0:
            raise ValueError('Data classes should contain at least one class!')
        model_classes = data_classes.copy()
    elif adapt_type == 'MERGE':
        model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
    else:
        raise KeyError(f'{adapt_type} is not supported for task_adapt options!')
    if task_adapt_type == 'mpa':
        data_classes = model_classes
    cfg.task_adapt.final = model_classes
    cfg.model.task_adapt = ConfigDict(
        src_classes=org_model_classes,
        dst_classes=model_classes,
    )

    # Model architecture
    if 'roi_head' in cfg.model:
        # For Faster-RCNNs
        cfg.model.roi_head.bbox_head.num_classes = len(model_classes)
    else:
        # For other architectures (including SSD)
        cfg.model.bbox_head.num_classes = len(model_classes)

    # Pseudo label augmentation
    pre_stage_res = kwargs.get('pre_stage_res', None)
    tr_data_cfg = get_train_data_cfg(cfg)
    if pre_stage_res:
        logger.info(f'pre-stage dataset: {pre_stage_res}')
        tr_data_cfg.pre_stage_res = pre_stage_res
        if tr_data_cfg.type not in ['CocoDataset', 'PseudoIncrCocoDataset']:
            raise NotImplementedError(f'Pseudo label loading for {tr_data_cfg.type} is not yet supported!')
        tr_data_cfg.org_type = tr_data_cfg.type
        tr_data_cfg.type = 'PseudoIncrCocoDataset'
        if 'pseudo_threshold' in cfg.hparams:
            logger.info(f'Setting pseudo threshold: {cfg.hparams.pseudo_threshold}')
            tr_data_cfg.pseudo_threshold = cfg.hparams.pseudo_threshold
        data_classes = model_classes  # assuming pseudo classes + data classes == model classes

    # Data pipeline
    if data_classes != model_classes:
        # Trying to alter class indices of training data according to model class order
        class_adapt_cfg = dict(type='AdaptClassLabels', src_classes=data_classes, dst_classes=model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, op in enumerate(pipeline_cfg):
            if op['type'] == 'LoadAnnotations':  # insert just after this op
                op_next_ann = pipeline_cfg[i + 1]
                if op_next_ann['type'] == class_adapt_cfg['type']:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)

    # Evaluation dataset
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

    # Training hook for task adaptation
    task_adapt_hook = ConfigDict(
        type='TaskAdaptHook',
        src_classes=org_model_classes,
        dst_classes=model_classes,
        model_type=cfg.model.type,
    )
    update_or_add_custom_hook(cfg, task_adapt_hook)

    # Anchor setting
    if cfg['task_adapt'].get('use_mpa_anchor', False):
        configure_anchor(cfg)

    # Incremental learning
    if task_adapt_type == 'mpa' and cfg.model.bbox_head.type != 'PseudoSSDHead':
        tr_data_cfg.img_ids_dict = get_img_ids_for_incr(cfg, org_model_classes, model_classes)
        if tr_data_cfg.type not in ['CocoDataset', 'DetIncrCocoDataset']:
            raise NotImplementedError(f'Pseudo label loading for {tr_data_cfg.type} is not yet supported!')
        tr_data_cfg.org_type = tr_data_cfg.type
        tr_data_cfg.type = 'DetIncrCocoDataset'
        if cfg.model.bbox_head.type in ['SSDHead', 'CustomSSDHead']:
            gamma = 1 if cfg['task_adapt'].get('efficient_mode', False) else 2
            cfg.model.bbox_head.type = 'CustomSSDHead'
            cfg.model.bbox_head.loss_cls = ConfigDict(
                type='FocalLoss',
                loss_weight=1.0,
                gamma=gamma,
                reduction='none',
            )
        elif cfg.model.bbox_head.type in ['ATSSHead']:
            gamma = 3 if cfg['task_adapt'].get('efficient_mode', False) else 4.5
            cfg.model.bbox_head.loss_cls.gamma = gamma
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type='TaskAdaptHook',
                sampler_flag=True,
                efficient_mode=cfg['task_adapt'].get('efficient_mode', False)
            )
        )
        update_or_add_custom_hook(cfg, ConfigDict(type='EMAHook'))


def configure_regularization(cfg):
    logger = get_root_logger()
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


def get_model_anchor_ratio(cfg):
    """Extract model anchor ratio info from checkpoint file.
    """
    anchor_ratio = []
    ckpt_path = cfg.get('load_from', None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        meta = ckpt.get('meta', {})
        anchor_ratio = np.array(meta.get('anchor_ratio', []))
    if len(anchor_ratio) == 0:
        anchor_ratio = np.array([1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0])
    return anchor_ratio


def get_img_ids_for_incr(cfg, org_model_classes, model_classes):
    # get image ids of old classes & new class
    # to setup experimental dataset (COCO format)
    new_classes = np.setdiff1d(model_classes, org_model_classes).tolist()
    old_classes = np.intersect1d(org_model_classes, model_classes).tolist()

    src_data_cfg = get_train_data_cfg(cfg)
    data_cfg = cfg.data.test.copy()
    data_cfg.test_mode = src_data_cfg.get('test_mode', False)
    data_cfg.ann_file = src_data_cfg.ann_file
    data_cfg.img_prefix = src_data_cfg.img_prefix

    old_data_cfg = data_cfg.copy()
    old_data_cfg.classes = old_classes
    old_dataset = build_dataset(old_data_cfg)

    ids_old = old_dataset.dataset.img_ids

    rng = np.random.default_rng(54321)
    if len(new_classes) > 0:
        data_cfg.classes = new_classes
        dataset = build_dataset(data_cfg)
        ids_new = dataset.dataset.img_ids
        ids_inter = np.intersect1d(ids_old, ids_new)
        if len(ids_inter) > 0:
            ids_new_ = ids_inter  # Assuming new class annotation is added to existing images
        else:
            ids_new_ = ids_new  # Assuming new class annotation is added to new images
        num_new_samples = cfg.task_adapt.get('num_new_data_samples', 100)
        sampled_ids_new = rng.permutation(ids_new_)[0:num_new_samples].tolist()
    else:
        sampled_ids_new = []

    ids_old = np.setdiff1d(ids_old, sampled_ids_new).tolist()
    sampled_ids = ids_old + sampled_ids_new
    outputs = dict(
        old_classes=old_classes,
        new_classes=new_classes,
        img_ids=sampled_ids,
        img_ids_old=ids_old,
        img_ids_new=sampled_ids_new,
    )
    return outputs


def configure_hyperparams(cfg, training, **kwargs):
    hyperparams = kwargs.get('hyperparams', None)
    if hyperparams is not None:
        bs = hyperparams.get('bs', None)
        if bs is not None:
            cfg.data.samples_per_gpu = bs

        lr = hyperparams.get('lr', None)
        if lr is not None:
            cfg.optimizer.lr = lr


def update_or_add_custom_hook(cfg, hook):
    """Update hook cfg if same type is in custom_hook or append it
    """
    custom_hooks = cfg.get('custom_hooks', [])
    custom_hooks_updated = False
    for custom_hook in custom_hooks:
        if custom_hook['type'] == hook['type']:
            custom_hook.update(hook)
            custom_hooks_updated = True
            break
    if not custom_hooks_updated:
        custom_hooks.append(hook)
    cfg['custom_hooks'] = custom_hooks
