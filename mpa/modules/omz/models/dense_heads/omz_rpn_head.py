import torch.nn.functional as F
# from mmcv.ops import batched_nms

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_head import RPNHead

from mpa.modules.omz.utils.omz_utils import get_net
from mpa.modules.omz.utils.omz_to_torch import gen_convolution


@HEADS.register_module()
class OmzRPNHead(RPNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, in_channels, model_path, rpn_conv_name, relu_type, rpn_cls_name, rpn_reg_name, **kwargs):
        self.model_path = model_path
        self.rpn_conv_name = rpn_conv_name
        self.relu_type = relu_type
        self.rpn_cls_name = rpn_cls_name
        self.rpn_reg_name = rpn_reg_name
        super(OmzRPNHead, self).__init__(in_channels, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        omz_net = get_net(self.model_path)
        self.rpn_conv = gen_convolution(omz_net.layers[self.rpn_conv_name], training=True)
        self.rpn_cls = gen_convolution(omz_net.layers[self.rpn_cls_name], training=True)
        self.rpn_reg = gen_convolution(omz_net.layers[self.rpn_reg_name], training=True)

    def init_weights(self):
        pass

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        if self.relu_type == 'relu6':
            x = F.relu6(x, inplace=True)
        else:
            x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(OmzRPNHead, self).train(mode)
        if mode:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
