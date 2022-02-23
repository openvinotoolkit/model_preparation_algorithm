import os
import time
import torch
from mmcv import ConfigDict
from mmseg.utils import get_root_logger

from mpa.stage import Stage


class SegStage(Stage):
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
            model_task = cfg.model.pop('task', 'segmentation')
            if model_task != 'segmentation':
                raise ValueError(
                    f'Given model_cfg ({model_cfg.filename}) is not supported by segmentation recipe'
                )
        self._configure_model(cfg, training, **kwargs)

        if not cfg.get('task_adapt'):   # if task_adapt dict is empty, just pop to pass task_adapt
            cfg.pop('task_adapt')

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

        # if training:
        #    if cfg.data.get('val', False):
        #        self.validate = True

        # Task
        if 'task_adapt' in cfg:
            self._configure_task(cfg, training, **kwargs)

        # Other hyper-parameters
        if 'hyperparams' in cfg:
            self._configure_hyperparams(cfg, training, **kwargs)

        return cfg

    def _init_logger(self):
        ''' override to initalize mmdet logger instead of mpa one.
        '''
        if self.logger is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            work_dir = os.path.dirname(self.cfg.work_dir)
            # print(f'workdir for the logger of segmentation tasks: {work_dir}')
            self.logger = get_root_logger(log_file=os.path.join(work_dir, f'{timestamp}.log'),
                                          log_level=self.cfg.log_level)

    def _get_model_classes(self, cfg):
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
        return list(classes)

    def _get_data_classes(self, cfg):
        # TODO: getting from actual dataset
        return list(get_train_data_cfg(cfg).get('classes', []))

    def _configure_model(self, cfg, training, **kwargs):
        pass

    def _configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation
        """
        # Current : It doesn't matter if the dataset is TaskAdaptDatasetSeg.
        # TODO : ONLY TaskAdaptDatasetSeg is allowed in task_adapt.
        self.logger = get_root_logger()
        self.logger.info(f'task config!!!!: training={training}')

        adapt_type = cfg['task_adapt'].get('op', 'REPLACE')
        org_model_classes = self._get_model_classes(cfg)
        data_classes = self._get_data_classes(cfg)

        # Model classes
        if adapt_type == 'REPLACE':
            if len(data_classes) == 0:
                raise ValueError('Data classes should contain at least one class!')
            model_classes = data_classes.copy()
        elif adapt_type == 'MERGE':
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f'{adapt_type} is not supported for task_adapt options!')

        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        # Model architecture
        if 'decode_head' in cfg.model:
            for head in cfg.model.decode_head:
                head.num_classes = len(model_classes)

        # Dataset
        for mode in ['train', 'val', 'test']:
            if cfg.data[mode]['type'] == 'TaskAdaptDatasetSeg':
                cfg.data[mode]['model_classes'] = model_classes
            else:
                # Wrap original dataset config
                org_type = cfg.data[mode]['type']
                cfg.data[mode]['type'] = 'TaskAdaptDatasetSeg'
                cfg.data[mode]['org_type'] = org_type
                cfg.data[mode]['model_classes'] = model_classes

    def _configure_hyperparams(self, cfg, training, **kwargs):
        hyperparams = kwargs.get('hyperparams', None)
        if hyperparams is not None:
            bs = hyperparams.get('bs', None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get('lr', None)
            if lr is not None:
                cfg.optimizer.lr = lr


def get_train_data_cfg(cfg):
    if 'dataset' in cfg.data.train:  # Concat|RepeatDataset
        return cfg.data.train.dataset
    else:
        return cfg.data.train
