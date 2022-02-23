import os
from os import path as osp
import torch.onnx
from functools import partial

from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmcls.models import build_classifier
from mpa.registry import STAGES
from .stage import ClsStage

from mpa.utils import logger


@STAGES.register_module()
class ClsExporter(ClsStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run exporter stage

        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            logger.warning(f'mode for this stage {mode}')
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        onnx_path = osp.join(cfg.work_dir, "classifier.onnx")

        # build the model and load checkpoint
        model = build_classifier(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        print('load checkpoint from ' + cfg.load_from)
        _ = load_checkpoint(model, cfg.load_from, map_location='cpu')
        model.eval()
        model.forward = partial(model.forward, return_loss=False)
        x = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        torch.onnx.export(model,
                          x,
                          onnx_path,
                          export_params=True,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

        model_name = 'model_fp16'
        input_shape = '[1,3,224,224]'

        mo_args = {
            'input_model': onnx_path,
            'data_type': 'FP16',
            'input_shape': input_shape,
            'log_level': 'WARNING',
            'model_name': model_name
        }

        from mpa.utils import mo_wrapper
        ret, msg = mo_wrapper.generate_ir(cfg.work_dir, cfg.work_dir, silent=True, **mo_args)
        os.remove(onnx_path)
