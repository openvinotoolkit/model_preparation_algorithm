import os.path as osp
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmcls.datasets import build_dataloader as build_dataloader_cls
from mmcls.datasets import build_dataset as build_dataset_cls
from mmcls.models import build_classifier
from ..stage import Stage
from . import logger
from ..registry import STAGES
from mmdet.models import build_detector
from mmdet.datasets import build_dataloader as build_dataloader_det
from mmdet.datasets import build_dataset as build_dataset_det
from ..det.stage import DetectionStage, get_train_data_cfg
from ..det.inferrer import replace_ImageToTensor
from ..cls.stage import ClsStage
from torchvision.transforms import ToPILImage, ToTensor, Resize
from mmcls.datasets import PIPELINES as PIPELINES_mmcls
from mmdet.datasets import PIPELINES as PIPELINES_mmdet
from mda import MDA

TASK_STAGE = {'classification': ClsStage, 'detection': DetectionStage}


def fix_dataset_size(results):
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    resize = Resize([128, 128])
    results = to_tensor(resize(to_pil(results)))

    return results


@PIPELINES_mmcls.register_module()
class ClsFixDatasetSize(object):
    def __call__(self, results):
        results['img'] = fix_dataset_size(results['img'])
        return results


@PIPELINES_mmdet.register_module()
class DetFixDatasetSize(object):
    def __call__(self, results):
        results['img'][0] = fix_dataset_size(results['img'][0])
        return results


@STAGES.register_module()
class MdaRunner(Stage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = kwargs.get('task', None)  # classification or detection
        if self.task not in ['classification', 'detection']:
            raise ValueError(f'task shoulde be "classification" or "detection". Current value is {self.task}')
        self.task_stage = TASK_STAGE[self.task](**kwargs)

        mda_metric = kwargs.get('mda_metric', 'z-score')  # 'z-score', 'cos-sim', 'kl' or 'wst'
        if mda_metric not in ['z-score', 'cos-sim', 'kl', 'wst']:
            raise ValueError(f'mda_metric should be "z-score", "cos-sim", "kl" or "wst". \
                Current value is "{mda_metric}"')
        self.mda_metric = mda_metric

    def make_dataset(self, cfg):
        if self.task == "classification":
            input_source = cfg.get('input_source', 'test')
            print(f'MDA on input source: data.{input_source}')
            cfg.data[input_source]['pipeline'].append({'type': 'ClsFixDatasetSize'})
            dataset = build_dataset_cls(cfg.data[input_source])
        elif self.task == "detection":
            input_source = cfg.get('input_source', 'test')
            # Current version of OTE detection supports only batch size 1 during evaluation and inference.
            # When version is updated, this code will be fixed.
            samples_per_gpu = 1

            # Input source
            input_source = cfg.get('input_source', 'test')
            print(f'MDA on input source: data.{input_source}')
            if input_source == 'train':
                src_data_cfg = get_train_data_cfg(cfg)
            else:
                src_data_cfg = cfg.data[input_source]

            if samples_per_gpu > 1:  # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                src_data_cfg.pipeline = replace_ImageToTensor(src_data_cfg.pipeline)

            data_cfg = src_data_cfg.copy()
            dataset = build_dataset_det(data_cfg)

        return dataset

    def make_model(self, cfg):
        if self.task == "classification":
            # build the model and load checkpoint
            model = build_classifier(cfg.model)
        elif self.task == "detection":
            # Target classes
            if 'task_adapt' in cfg:
                target_classes = cfg.task_adapt.final
            else:
                target_classes = self.dataset.CLASSES

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

            train_cfg = cfg.get('train_cfg', None)
            test_cfg = cfg.get('test_cfg', None)
            model = build_detector(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

            model.CLASSES = target_classes

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        return model

    def analyze_model_drift(self, cfg):
        self.dataset = self.make_dataset(cfg)

        # Data loader
        if self.task == "classification":
            data_loader = build_dataloader_cls(
                self.dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False,
                round_up=False)
        elif self.task == "detection":
            data_loader = build_dataloader_det(
                self.dataset,
                samples_per_gpu=cfg.data.test.get('samples_per_gpu', 1),
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False)

        self.model = self.make_model(cfg)
        self.model = MMDataParallel(self.model, device_ids=[0])
        mda = MDA(self.mda_metric, verbose=1, mode='stable')
        output = mda.measure(self.model, data_loader)

        print(f'Model drift score is {output}')
        with open(osp.join(osp.abspath(cfg.work_dir), 'mpa_output.txt'), 'w') as f:
            f.write(str(output))

        return output

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage

        - Run inference
        - Run evaluation via MMDetection -> MMCV
        """
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            logger.warning(f'mode for this stage {mode}')
            return {}

        self.task_stage._init_logger()
        cfg = self.task_stage.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        mda_results = self.analyze_model_drift(cfg)

        return mda_results
