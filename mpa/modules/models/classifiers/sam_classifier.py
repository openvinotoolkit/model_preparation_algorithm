# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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
        self.multilabel = kwargs.pop('multilabel',False)
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

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.mixup is not None:
            img, gt_label = self.mixup(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        if kwargs.get('img_metas', False) and len(kwargs['img_metas'][-1]['ignored_labels']) > 0:
            loss = self.head.forward_train(x, gt_label, **kwargs)
        else:
            loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect model as output state_dict for OTE model compatibility
        """
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ['OTEMobileNetV3', 'OTEEfficientNet', 'OTEEfficientNetV2']:
            return

        output = OrderedDict()
        if backbone_type == 'OTEMobileNetV3':
            for k, v in state_dict.items():
                if k.startswith('backbone'):
                    k = k.replace('backbone.', '')
                elif k.startswith('head'):
                    k = k.replace('head.', '')
                    if '3' in k:  # MPA uses "classifier.3", OTE uses "classifier.4". Convert for OTE compatibility.
                        k = k.replace('3', '4')
                        if module.multilabel and not module.is_export:
                            v = v.t()
                output[k] = v

        elif backbone_type == 'OTEEfficientNet':
            for k, v in state_dict.items():
                if k.startswith('backbone'):
                    k = k.replace('backbone.', '')
                elif k.startswith('head'):
                    k = k.replace('head', 'output')
                    if module.multilabel and not module.is_export:
                        k = k.replace('fc', 'asl')
                        v = v.t()
                output[k] = v

        elif backbone_type == 'OTEEfficientNetV2':
            for k, v in state_dict.items():
                if k.startswith('backbone'):
                    k = k.replace('backbone.', '')
                elif k == 'head.fc.weight':
                    k = k.replace('head.fc', 'model.classifier')
                    if not module.is_export:
                        v = v.t()
                output[k] = v

        return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to model for OTE model compatibility
        """
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ['OTEMobileNetV3', 'OTEEfficientNet', 'OTEEfficientNetV2']:
            return

        if backbone_type == 'OTEMobileNetV3':
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith('classifier.'):
                    if '4' in k:
                        k = 'head.'+k.replace('4', '3')
                        if module.multilabel:
                            v = v.t()
                    else:
                        k = 'head.'+k
                elif not k.startswith('backbone.'):
                    k = 'backbone.'+k
                state_dict[k] = v

        elif backbone_type == 'OTEEfficientNet':
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith('features.') and 'activ' not in k:
                    k = 'backbone.'+k
                elif k.startswith('output.'):
                    k = k.replace('output', 'head')
                    if module.multilabel:
                        k = k.replace('asl', 'fc')
                        v = v.t()
                state_dict[k] = v

        elif backbone_type == 'OTEEfficientNetV2':
            for k in list(state_dict.keys()):
                v = state_dict.pop(k)
                if k.startswith('model.classifier'):
                    k = k.replace('model.classifier', 'head.fc')
                    v = v.t()
                elif k.startswith('model'):
                    k = 'backbone.'+k
                state_dict[k] = v
        else:
            logger.info('conversion is not required.')

    @staticmethod
    def load_state_dict_mixing_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading
        """
        backbone_type = type(model.backbone).__name__
        if backbone_type not in ['OTEMobileNetV3', 'OTEEfficientNet', 'OTEEfficientNetV2']:
            return

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f'{chkpt_classes} -> {model_classes} ({model2chkpt})')
        model_dict = model.state_dict()

        if backbone_type == 'OTEMobileNetV3':
            if model.multilabel:
                param_names = ['classifier.4.weight']
            else:
                param_names = ['classifier.4.weight', 'classifier.4.bias']

        elif backbone_type == 'OTEEfficientNet':
            if model.multilabel:
                param_names = ['output.asl.weight']
            else:
                param_names = ['output.fc.weight']
                if 'head.fc.bias' in chkpt_dict.keys():
                    param_names.append('output.fc.bias')

        elif backbone_type == 'OTEEfficientNetV2':
            param_names = [
                'model.classifier.weight',
            ]
            if 'head.fc.bias' in chkpt_dict.keys():
                param_names.append('head.fc.bias')

        for model_name in param_names:
            model_param = model_dict[model_name].clone()
            if backbone_type == 'OTEMobileNetV3':
                chkpt_name = 'head.'+model_name.replace('4', '3')
                if model.multilabel:
                    model_param = model_param.t()
            elif backbone_type in 'OTEEfficientNet':
                if model.multilabel:
                    chkpt_name = model_name.replace('output.asl.', 'head.fc.')
                    model_param = model_param.t()
                else:
                    chkpt_name = model_name.replace('output', 'head')

            elif backbone_type in 'OTEEfficientNetV2':
                if model_name.endswith('bias'):
                    chkpt_name = model_name
                else:
                    chkpt_name = model_name.replace('model.classifier', 'head.fc')
                    model_param = model_param.t()

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
