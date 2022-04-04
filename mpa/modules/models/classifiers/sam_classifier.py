from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models.classifiers.image import ImageClassifier
from mpa.modules.utils.task_adapt import map_class_names
from mpa.utils.logger import get_logger
from collections import OrderedDict
import functools

logger = get_logger()


@CLASSIFIERS.register_module()
class SAMClassifier(BaseClassifier):
    """SAM-enabled BaseClassifier"""

    def train_step(self, data, optimizer):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer)


@CLASSIFIERS.register_module()
class SAMImageClassifier(ImageClassifier):
    """SAM-enabled ImageClassifier"""

    def __init__(self, task_adapt=None, **kwargs):
        super().__init__(**kwargs)
        self.is_export = False
        self.featuremap = None
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(
            functools.partial(self.load_state_dict_pre_hook, self)
        )
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_mixing_hook,
                    self,  # model
                    task_adapt['dst_classes'],  # model_classes
                    task_adapt['src_classes']   # chkpt_classes
                )
            )

    def train_step(self, data, optimizer):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer)

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect model as output state_dict for OTE model compatibility
        """
        logger.info('----------------- SAMImageClassifier.state_dict_hook() called')
        if type(module.backbone).__name__ == 'OTEMobileNetV3':
            output = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('head.classifier'):
                    k = k.replace('head.', '')
                    if k.startswith('classifier.3'):
                        k = k.replace('classifier.3', 'classifier.4')
                elif k.startswith('backbone'):
                    k = k.replace('backbone.', '')
                output[k] = v
            return output

        elif type(module.backbone).__name__ == 'OTEEfficientNet':
            output = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone'):
                    k = k.replace('backbone.', '')
                elif k == 'head.fc.weight':
                    k = k.replace('head.fc', 'output.asl')
                    if not module.is_export:
                        v = v.t()
                output[k] = v
            return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to model for OTE model compatibility
        """
        logger.info('----------------- SAMImageClassifier.load_state_dict_pre_hook() called')
        if type(module.backbone).__name__ == 'OTEMobileNetV3':
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith('classifier.') or k.startswith('head.classifier'):
                    k = k.replace('head.', '')
                    if k.startswith('classifier.4'):
                        k = k.replace('classifier.4', 'classifier.3')
                        state_dict['head.'+k] = v
                    else:
                        state_dict['head.'+k] = v
                elif not k.startswith('backbone.') and not k.startswith('head.'):
                    state_dict['backbone.'+k] = v
                else:
                    state_dict[k] = v

        elif type(module.backbone).__name__ == 'OTEEfficientNet':
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith('features.') and 'activ' not in k:
                    state_dict['backbone.'+k] = v
                elif k == 'output.asl.weight':
                    k = k.replace('output.asl', 'head.fc')
                    state_dict[k] = v.t()
                else:
                    state_dict[k] = v
        else:
            logger.info('conversion is not required.')

    @staticmethod
    def load_state_dict_mixing_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading
        """
        logger.info(f'----------------- SAMImageClassifier.load_state_dict_pre_hook() called w/ prefix: {prefix}')

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f'{chkpt_classes} -> {model_classes} ({model2chkpt})')

        model_dict = model.state_dict()
        backbone_type = type(model.backbone).__name__
        if backbone_type == 'OTEMobileNetV3':
            param_names = [
                'classifier.4.weight',
                'classifier.4.bias',
            ]
        elif backbone_type == 'OTEEfficientNet':
            param_names = [
                'output.asl.weight',
            ]

        for model_name in param_names:
            if backbone_type == 'OTEMobileNetV3':
                chkpt_name = 'head.'+model_name.replace('4', '3')
                model_param = model_dict[model_name].clone()
            elif backbone_type == 'OTEEfficientNet':
                chkpt_name = 'head.fc.weight'
                model_param = model_dict[model_name].clone().t()

            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f'Skipping weight copy: {chkpt_name}')
                continue

            # Mix weights
            chkpt_param = chkpt_dict[chkpt_name]
            for m, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[m].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
           Overriding for OpenVINO export with features
        """
        x = self.featuremap = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_metas):
        """Test without augmentation.
           Overriding for OpenVINO export with features
        """
        x = self.extract_feat(img)
        logits = self.head.simple_test(x)
        if self.is_export:
            return logits, self.featuremap, x  # (logits, featuremap, vector)
        else:
            return logits
