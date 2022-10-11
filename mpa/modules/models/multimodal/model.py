from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist

from mpa.multimodal.builder import MODELS
from mpa.multimodal.builder import build_backbone, build_head
from mpa.utils.logger import get_logger

logger = get_logger()

@MODELS.register_module()
class MultimodalModel(nn.Module):
    def __init__(self,
                 task,
                 visual_encoder,
                 textual_encoder,
                 tabular_encoder,
                 head,
                 pretrained=None,
                 **kwargs):
        super(MultimodalModel, self).__init__()
        self.task = task
        self.logger = logger
        self.modalities = kwargs.get('modalities')
        
        ### build visual part
        if 'vision' in self.modalities:
            self.visual_encoder = build_backbone(visual_encoder)
        
        ### build textual part
        if 'text' in self.modalities:
            self.textual_encoder = build_backbone(textual_encoder)
        
        ### build tabular part
        if 'tabular' in self.modalities:
            self.tabular_encoder = build_backbone(tabular_encoder)

        ### build heads
        head.modalities = self.modalities
        self.head = build_head(head)
        
        self._init_weights(pretrained=pretrained)
    
    def _init_weights(self, pretrained=None):
        pass
    
    def forward(self, return_loss=True, **data):
        feature_dict = {}
        if hasattr(self, 'visual_encoder'):
            img_feature = self.visual_encoder(data['img'])
            feature_dict['vision_feature'] = img_feature
        if hasattr(self, 'texture_encoder'):
            text_feature = self.textual_encoder()
            feature_dict['textual_feature'] = text_feature
        if hasattr(self, 'tabular_encoder'):
            tabular_feature = self.tabular_encoder(data['meta_info'])
            feature_dict['tabular_feature'] = tabular_feature

        if return_loss:
            return self.forward_train(feature_dict, data['gt_label'])
        else:
            return self.forward_test(feature_dict)
        
    def forward_train(self, feature_dict, gt_label, **kwargs):
        return dict(loss=self.head(feature_dict, gt_label))

    def forward_test(self, feature_dict, **kwargs):
        return self.head(feature_dict)
        
    def train_step(self, data, optimizer):
        """_summary_

        Args:
            data (list): img, target, meta

        Returns:
            _type_: _description_
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=data['gt_label'].shape[0]
        )
        
        return outputs
        

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
