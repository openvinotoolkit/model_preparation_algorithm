import os.path as osp
import numpy as np
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector

from mpa.registry import STAGES
from .stage import DetectionStage, get_train_data_cfg


@STAGES.register_module()
class DetectionInferrer(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage for detection

        - Configuration
        - Environment setup
        - Run inference via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        self.logger.info('infer!')

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        outputs = self.infer(cfg)

        # Save outputs
        output_file_path = osp.join(cfg.work_dir, 'pre_stage_res.npy')
        np.save(output_file_path, outputs, allow_pickle=True)
        return dict(pre_stage_res=output_file_path)

    def infer(self, cfg):
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # Input source
        input_source = cfg.get('input_source', 'test')
        self.logger.info(f'Inferring on input source: data.{input_source}')
        if input_source == 'train':
            src_data_cfg = get_train_data_cfg(cfg)
        else:
            src_data_cfg = cfg.data[input_source]
        data_cfg = cfg.data.test.copy()
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
        if 'task_adapt' in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(f'target_classes={target_classes} is empty check the metadata from model ckpt or recipe '
                               f'configuration')
        else:
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

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        # Checkpoint
        if cfg.get('load_from', None):
            load_checkpoint(model, cfg.load_from, map_location='cpu')

        # Inference
        model = MMDataParallel(model, device_ids=[0])
        detections = single_gpu_test(model, data_loader)
        outputs = dict(
            config=cfg.pretty_text,
            classes=target_classes,
            detections=detections
        )

        return outputs
