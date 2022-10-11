import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init
from mmcv.cnn import build_norm_layer

from mpa.multimodal.builder import HEADS, build_loss
from mmcls.models.losses import Accuracy

from mpa.utils.logger import get_logger

logger = get_logger()

def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    if init_linear not in ['normal', 'kaiming']:
        raise ValueError("Undefined init_linear: {}".format(init_linear))
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

@HEADS.register_module()
class MultimodalClsHead(nn.Module):
    """The Classification head for multimodal classification
    Args:
        nn (_type_): _description_
    """
    
    def __init__(self,
                 n_classes,
                 vision_feature_channels=None,
                 textual_feature_in_dim=None,
                 tabular_feature_in_dim=None,
                 out_dim_per_modal=256,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1,5),
                 **kwargs,
                 ):
        super(MultimodalClsHead, self).__init__()
        self.topk = topk
        self.modalities = kwargs.get('modalities')
        
        if 'vision' in self.modalities:
            self.vision_linear = nn.Linear(vision_feature_channels, out_dim_per_modal)
        if 'text' in self.modalities:
            self.textual_linear = nn.Linear(textual_feature_in_dim, out_dim_per_modal)
        if 'tabular' in self.modalities:
            self.tabular_linear = nn.Linear(tabular_feature_in_dim, out_dim_per_modal)

        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.fc = nn.Linear(out_dim_per_modal*len(self.modalities), n_classes)
    
    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        losses['loss'] = loss
        return losses
        
    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)
    
    def forward(self, x, gt_label=None):
        if isinstance(x, dict):
            cat_vector = None
            for i, (k, v) in enumerate(x.items()):
                if k is 'vision_feature':
                    avg_pool = nn.AvgPool2d(v.shape[-1])
                    feature = self.vision_linear(avg_pool(v).view(v.shape[0],-1))
                elif k is 'textual_feature':
                    feature = self.textual_linear(v)
                elif k is 'tabular_feature':
                    feature = self.tabular_linear(v)
                else:
                    raise NotImplementedError("{} is not supported yet".format(k))
                
                if i == 0:
                    cat_vector = feature
                else:
                    cat_vector = torch.cat((cat_vector, feature), dim=1)

            cls_score = self.fc(cat_vector)
            if gt_label is not None:
                losses = self.loss(cls_score, gt_label.view(gt_label.shape[0],))
                return losses
            else:
                pred = F.softmax(cls_score, dim=1)
                return list(pred.detach().cpu().numpy())
        else:
            raise NotImplementedError('Expected type of input is dict, but got {}.'.format(type(x)))


@HEADS.register_module()
class MultimodalRegHead(nn.Module):
    """The Regression head for multimodal classification
    """
    
    def __init__(self,
                 vision_feature_channels=None,
                 textual_feature_in_dim=None,
                 tabular_feature_in_dim=None,
                 out_dim_per_modal=256,
                 loss=dict(type='RMSELoss', loss_weight=1.0),
                 **kwargs,
                 ):
        super(MultimodalRegHead, self).__init__()
        self.modalities = kwargs.get('modalities')
        
        if 'vision' in self.modalities:
            self.vision_linear = nn.Linear(vision_feature_channels, out_dim_per_modal)
        if 'text' in self.modalities:
            self.textual_linear = nn.Linear(textual_feature_in_dim, out_dim_per_modal)
        if 'tabular' in self.modalities:
            self.tabular_linear = nn.Linear(tabular_feature_in_dim, out_dim_per_modal)

        self.compute_loss = build_loss(loss)
        self.fc = nn.Linear(out_dim_per_modal * len(self.modalities), 1)
    
    def loss(self, out, gt_label):
        batch_size = out.shape[0]
        # compute loss
        loss = self.compute_loss(out, gt_label, avg_factor=batch_size)
        return {'loss': loss}
        
    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)
    
    def forward(self, x, gt_label=None):
        if isinstance(x, dict):
            cat_vector = None
            for i, (k, v) in enumerate(x.items()):
                if k is 'vision_feature':
                    avg_pool = nn.AvgPool2d(v.shape[-1])
                    feature = self.vision_linear(avg_pool(v).view(v.shape[0],-1))
                elif k is 'textual_feature':
                    feature = self.textual_linear(v)
                elif k is 'tabular_feature':
                    feature = self.tabular_linear(v)
                else:
                    raise NotImplementedError("{} is not supported yet".format(k))
                
                if i == 0:
                    cat_vector = feature
                else:
                    cat_vector = torch.cat((cat_vector, feature), dim=1)

            out = self.fc(cat_vector)
            if gt_label is not None:
                losses = self.loss(out, gt_label.view(gt_label.shape[0],))
                return losses
            else:
                return list(out.detach().cpu().numpy())
        else:
            raise NotImplementedError('Expected type of input is dict, but got {}.'.format(type(x)))