import torch
# from torch import nn

# import tlt.core.base.model
from mpa.modules.omz.utils.omz_to_torch import OMZModel, generate_moduledict
from mpa.modules.omz.utils.omz_utils import get_net, get_layers_list
# from tlt.core.base.builder import Builder
# from tlt.core.base.model import Backbone


# @Builder.register
class OmzBackbone(OMZModel):
    def __init__(self, mode, model_path, last_layer_name, normalized_img_input, **kwargs):
        moduledict, layers_info, parents_info = \
            self._parse_model_xml(model_path, last_layer_name, mode == 'train',
                                  mode == 'export', normalized_img_input=normalized_img_input)
        super().__init__(
                moduledict=moduledict,
                layers_info=layers_info,
                parents_info=parents_info,
                **kwargs)
        self.feature_dims = []

    @staticmethod
    def _parse_model_xml(model_path, last_layer_name, training, exporting, normalized_img_input):
        omz_net = get_net(model_path)
        layers = get_layers_list(omz_net.layers, omz_net.inputs, omz_net.outputs, 'all')
        module_dict, _, layers_dict, parents_dict = generate_moduledict(
                layers, num_classes=0, training=training, is_export=exporting,
                last_layer=last_layer_name, normalized_img_input=normalized_img_input)
        return module_dict, layers_dict, parents_dict

    def forward(self, *inputs):
        # inputs = inputs*255  # OMZ models take 0~255 inputs
        feat = super().forward(inputs)
        if len(self.feature_dims) == 0:
            self.feature_dims = [feature.shape[1] for feature in self.featuredict.values()]
        return feat

    def get_feature_dims(self):
        if len(self.feature_dims) == 0:
            with torch.no_grad():
                self.forward(torch.zeros(1, 3, 64, 64))
        return self.feature_dims


# @Builder.register
class OmzBackboneDummy(OmzBackbone):

    def forward(self, *inputs):
        image = inputs[0]
        dummy_im_info = torch.Tensor([1, 1, 1.0, 1.0])
        OMZModel.forward(self, [image, dummy_im_info])
        features = list(self.featuredict.values())
        if len(self.feature_dims) == 0:
            for feature in features:
                if len(feature.shape) > 1:
                    self.feature_dims.append(feature.shape[1])
        return features


# @Builder.register
class OmzBackboneFRCNN(OmzBackbone):

    def forward(self, *inputs):
        image = inputs[0]
        dummy_im_info = torch.Tensor([1, 1, 1.0, 1.0])
        feat = OMZModel.forward(self, [image, dummy_im_info])
        features = list(self.featuredict.values())
        if len(self.feature_dims) == 0:
            for feature in features:
                if len(feature.shape) > 1:
                    self.feature_dims.append(feature.shape[1])
        return feat
