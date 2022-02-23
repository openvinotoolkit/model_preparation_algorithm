import os.path as osp
import mmcv

from mpa.registry import STAGES
from .inferrer import SegInferrer


@STAGES.register_module()
class SegEvaluator(SegInferrer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        self._init_logger()
        mode = kwargs.get('mode', 'eval')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        self.logger.info('evaluate!')

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Inference
        infer_results = super().infer(cfg)
        segmentations = infer_results['segmentations']

        # Evaluate inference results
        eval_kwargs = self.cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
            eval_kwargs.pop(key, None)
        eval_result = self.dataset.evaluate(segmentations, **eval_kwargs)
        self.logger.info(eval_result)

        return dict(mAP=eval_result.get('bbox_mAP_50', 0.0))
