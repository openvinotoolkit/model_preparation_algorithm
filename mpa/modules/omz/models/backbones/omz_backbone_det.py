# import tlt.core.base.model
# from tlt.core.omz.omz_to_torch import OMZModel
# from tlt.core.omz.utils.omz_util import OMZModel
# from tlt.core.base.builder import Builder
# from tlt.core.base.model import Backbone
# from tlt.core.backbone.omz import OmzBackboneFRCNN
from mpa.modules.omz.models.backbones.omz_backbone import OmzBackboneFRCNN

# from ..builder import BACKBONES
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class OmzBackboneDet(OmzBackboneFRCNN):
    def __init__(self,
                 mode,
                 model_path,
                 last_layer_name,
                 frozen_stages=-1,
                 norm_eval=True,
                 **kwargs):
        super().__init__(
                mode=mode,
                model_path=model_path,
                last_layer_name=last_layer_name,
                **kwargs)

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self._freeze_backbone()

    def forward(self, x):  # should return a tuple
        feature = super().forward(x)
        return tuple(feature)

    def init_weights(self, pretrained=None):
        pass

    def _freeze_backbone(self):
        if self.frozen_stages > 0:
            for m in self.model.values():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(OmzBackboneDet, self).train(mode)
        self._freeze_backbone()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
