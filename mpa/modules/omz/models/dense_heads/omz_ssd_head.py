import torch.nn as nn

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.ssd_head import SSDHead

from mpa.modules.omz.utils.omz_utils import get_net
from mpa.modules.omz.utils.omz_to_torch import generate_moduledict


@HEADS.register_module()
class OmzSSDHead(SSDHead):
    def __init__(self, model_path, reg_conv_name, cls_conv_name, org_input_size=0, **kwargs):
        super(OmzSSDHead, self).__init__(**kwargs)

        # TODO: workaround code to fit scales of anchors with omz model
        if org_input_size > 0:
            input_size = getattr(self.anchor_generator, 'input_size', org_input_size)
            if input_size != org_input_size:
                self.anchor_generator.base_sizes = [
                    bs*org_input_size/input_size for bs in self.anchor_generator.base_sizes
                ]
                self.anchor_generator.base_anchors = self.anchor_generator.gen_base_anchors()

        reg_convs = []
        cls_convs = []

        if isinstance(reg_conv_name, str):
            reg_conv_name = [reg_conv_name]
        if isinstance(cls_conv_name, str):
            cls_conv_name = [cls_conv_name]

        assert len(self.anchor_generator.num_base_anchors) == len(self.in_channels)
        assert len(reg_conv_name) == len(self.in_channels)
        assert len(cls_conv_name) == len(self.in_channels)

        omz_net = get_net(model_path)
        num_classes = self.num_classes + 1
        for i in range(len(self.in_channels)):
            in_channels = self.in_channels[i]
            num_anchors = self.anchor_generator.num_base_anchors[i]

            _reg_conv_name = reg_conv_name[i]
            _cls_conv_name = cls_conv_name[i]

            if isinstance(_reg_conv_name, str):
                _reg_conv_name = [_reg_conv_name]
            if isinstance(_cls_conv_name, str):
                _cls_conv_name = [_cls_conv_name]

            reg_layers = {k: omz_net.layers[k] for k in _reg_conv_name}
            cls_layers = {k: omz_net.layers[k] for k in _cls_conv_name}

            reg_module_dict, _, _, _ = generate_moduledict(reg_layers, num_classes)
            cls_module_dict, _, _, _ = generate_moduledict(cls_layers, num_classes)

            reg_module_dict.pop('gt_label')
            cls_module_dict.pop('gt_label')

            if reg_module_dict[_reg_conv_name[-1]].conv.out_channels == num_anchors*4:
                reg_convs.append(nn.Sequential(*reg_module_dict.values()))
            else:
                reg_convs.append(nn.Conv2d(in_channels, num_anchors*4, kernel_size=3, padding=1))

            if cls_module_dict[_cls_conv_name[-1]].conv.out_channels == num_anchors*num_classes:
                cls_convs.append(nn.Sequential(*cls_module_dict.values()))
            else:
                cls_convs.append(nn.Conv2d(in_channels, num_anchors*num_classes, kernel_size=3, padding=1))

        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

    def init_weights(self):
        pass
