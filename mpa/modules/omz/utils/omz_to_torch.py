import torch
from collections import OrderedDict
from torch import nn

# try:
#     from openvino import inference_engine as ie  # noqa: F401
#     from openvino.inference_engine import IENetwork, IEPlugin, IECore  # noqa: F401
# except Exception as e:
#     exception_type = type(e).__name__
#     print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
#     # sys.exit(1)

from mpa.modules.omz.custom_layers.base_layers import (Conv2dBN, Conv2dPadBN, AdaptiveBatchNorm2d, GAPooling,
                                                       MaxPool2DPad, AvgPool2DPad, Input, Label, Const, ScaleShift,
                                                       Eltwise, Concat, Reshape, Flatten, Crop, Power, Permute,
                                                       Interpolate)
from mpa.modules.omz.utils.omz_utils import get_last_backbone_layer


def gen_convolution(layer, training):
    params = layer.params
    in_channels = layer.in_data[0].shape[1]
    out_channels = int(params['output'])
    kernel_size = tuple(map(int, params['kernel'].split(',')))
    stride = tuple(map(int, params['strides'].split(',')))
    pads_begin = list(map(int, params['pads_begin'].split(',')))
    pads_end = list(map(int, params['pads_end'].split(',')))
    dilation = tuple(map(int, params['dilations'].split(',')))
    groups = int(params['group'])
    biases_val = layer.blobs.get('biases', None)
    auto_pad = params.get('auto_pad', None)

    if biases_val is None:
        bias = False
    else:
        bias = True

    if (layer.in_data[0].shape[2] == 1 and layer.in_data[0].shape[3] == 1):
        isBN = False
    else:
        isBN = True

    if (pads_begin[0] == pads_end[0]) and (pads_begin[1] == pads_end[1]):
        if auto_pad == 'valid':
            padding = tuple((0, 0))
        elif type(kernel_size) is not tuple and (auto_pad == 'same_upper' or auto_pad == 'same_lower'):
            padding = int((kernel_size-1)*0.5)
        else:
            padding = tuple((pads_begin[0], pads_begin[1]))
        convbn = Conv2dBN(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding,
                          dilation=dilation, groups=groups,
                          isbias=bias, isBN=isBN, training=training)
        weights = layer.blobs.get('weights', None)
        if weights is not None:
            weights = torch.reshape(torch.from_numpy(weights), convbn.weight.shape)
            convbn.weight.data = weights
        biases = layer.blobs.get('biases', None)
        if biases is not None:
            biases = torch.reshape(torch.from_numpy(biases), convbn.bias.shape)
            convbn.bias.data = biases
        return convbn
    else:
        if auto_pad == 'valid':
            padding = tuple((0, 0, 0, 0))
        else:
            padding = tuple((pads_begin[1], pads_end[1], pads_begin[0], pads_end[0]))
        convpadbn = Conv2dPadBN(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups,
                                isbias=bias, isBN=isBN, training=training)
        weights = layer.blobs.get('weights', None)
        if weights is not None:
            weights = torch.reshape(torch.from_numpy(weights), convpadbn.weight.shape)
            convpadbn.weight.data = weights
        biases = layer.blobs.get('biases', None)
        if biases is not None:
            biases = torch.reshape(torch.from_numpy(biases), convpadbn.bias.shape)
            convpadbn.bias.data = biases
        return convpadbn


def gen_batchnorm(layer, pl, training):
    num_features = layer.out_data[0].shape[1]
    if training:
        bn = AdaptiveBatchNorm2d(num_features=num_features, eps=1.0e-10).train()
    else:
        bn = AdaptiveBatchNorm2d(num_features=num_features, eps=1.0e-10).eval()
    weights = layer.blobs.get('weights', None)
    if weights is not None:
        weights = torch.reshape(torch.from_numpy(weights), bn.running_var.shape)
        bn.weight.data = weights
    biases = layer.blobs.get('biases', None)
    if biases is not None:
        biases = torch.reshape(torch.from_numpy(biases), bn.running_mean.shape)
        bn.bias.data = biases
    nn.init.ones_(bn.running_var)
    nn.init.zeros_(bn.running_mean)

    return bn


def gen_elu(layer):
    params = layer.params
    alpha = float(params['alpha'])
    return nn.ELU(alpha=alpha)


def gen_relu(layer):
    if 'negative_slope' in layer.params:
        return nn.LeakyReLU(float(layer.params['negative_slope']))
    else:
        return nn.ReLU()


def gen_fc(layer):
    in_channel = layer.in_data[0].shape[1]
    params = layer.params
    in_features = in_channel
    out_features = int(params['out-size'])
    fc = nn.Linear(in_features=in_features, out_features=out_features)
    weights = layer.blobs.get('weights', None)
    if weights is not None:
        weights = torch.reshape(torch.from_numpy(weights), fc.weight.shape)
        fc.weight.data = weights
    biases = layer.blobs.get('biases', None)
    if biases is not None:
        biases = torch.reshape(torch.from_numpy(biases), fc.bias.shape)
        fc.bias.data = biases

    return fc


def gen_pooling(layer):
    params = layer.params
    kernel_size = tuple(map(int, params['kernel'].split(',')))
    stride = tuple(map(int, params['strides'].split(',')))
    pads_begin = list(map(int, params['pads_begin'].split(',')))
    pads_end = list(map(int, params['pads_end'].split(',')))
    ceiling = True
    auto_pad = params.get('auto_pad', None)
    if params['rounding_type'] == 'floor':
        ceiling = False
    if (pads_begin[0] == pads_end[0]) and (pads_begin[1] == pads_end[1]):
        if auto_pad == 'valid':
            padding = tuple((0, 0))
        elif auto_pad == 'same_uppper' or auto_pad == 'same_lower':
            padding = int((kernel_size-1)*0.5)
        else:
            padding = tuple((pads_begin[1], pads_begin[0]))
        if params['pool-method'] == 'max':
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceiling)
        elif layer.out_data[0].shape[2] == layer.out_data[0].shape[3] and layer.out_data[0].shape[2] == 1:
            return GAPooling(stride, padding, ceiling)
        else:
            return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceiling)
    else:
        if auto_pad == 'valid':
            padding = tuple((0, 0, 0, 0))
        else:
            padding = tuple((pads_begin[1], pads_end[1], pads_begin[0], pads_end[0]))
        if params['pool-method'] == 'max':
            return MaxPool2DPad(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceiling)
        elif layer.out_data[0].shape[2] == layer.out_data[0].shape[3] and layer.out_data[0].shape[2] == 1:
            return GAPooling(stride, ceiling=ceiling)
        else:
            return AvgPool2DPad(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceiling)


def gen_softmax(layer):
    params = layer.params
    dim = int(params['axis'])
    return nn.Softmax(dim)


def gen_input(layer):
    dims = layer.out_data[0].shape
    return Input(dims)


def gen_label():
    return Label()


def gen_const(layer):
    data = layer.blobs['custom']
    return Const(data)


def gen_scaleshift(layer, is_trainable=False):
    dims = layer.out_data[0].shape
    ss = ScaleShift(dims, is_trainable)
    weights = layer.blobs.get('weights', None)
    if weights is not None:
        weights = torch.reshape(torch.from_numpy(weights), ss.weight.shape)
        ss.weight.data = weights
        # print('weights: ', weights)
    biases = layer.blobs.get('biases', None)
    if biases is not None:
        biases = torch.reshape(torch.from_numpy(biases), ss.bias.shape)
        ss.bias.data = biases
        # print('biases: ', biases)

    return ss


def gen_eltwise(layer):
    params = layer.params
    return Eltwise(params['operation'])


def gen_concat(layer):
    params = layer.params
    return Concat(int(params['axis']))


def gen_reshape(layer, parents_l=None, post_nms_topn=0):
    if len(layer.parents) > 1:
        blobs = parents_l.blobs
        dims = blobs['custom']
        return Reshape(dims, post_nms_topn=post_nms_topn)
    dims_str = layer.params.get('dim', None)
    if dims_str is not None:
        dims = list(dims_str.split(','))
        return Reshape(dims, from_const=False)
    else:
        return Flatten(1)


def gen_power(layer):
    params = layer.params
    power = float(params['power'])
    scale = float(params['scale'])
    shift = float(params['shift'])
    return Power(power, scale, shift)


def gen_crop(layer):
    params = layer.params
    axis = list(params['axis'].split(','))
    offset = list(params['offset'].split(','))
    dim = list(params['dim'].split(','))
    return Crop(axis, offset, dim)


def gen_sigmoid(layer):
    return nn.Sigmoid()


def gen_permute(layer):
    params = layer.params
    order = list(map(int, params['order'].split(',')))
    return Permute(order)


def gen_interpolate(layer, parents_l, is_export=False):
    params = layer.params
    align_corners = int(params['align_corners']) if not is_export else 0
    pad_beg = int(params['pad_beg'])
    pad_end = int(params['pad_end'])
    return Interpolate(align_corners, pad_beg, pad_end)


def gen_mvn(layer):
    in_channel = layer.in_data[0].shape[1]
    params = layer.params
    # across_channels = int(params['across_channels'])
    # normalize_variance = int(params['normalize_variance'])
    eps = float(params['eps'])
    # TODO: consider other parameters
    return nn.InstanceNorm2d(num_features=in_channel, eps=eps, affine=True)


def generate_moduledict(layers, num_classes, batch_size=32, training=True, is_export=False,
                        backbone_only=False, stop_layer=None, last_layer=None, normalized_img_input=False):
    moduledict = {}
    moduledict['gt_label'] = gen_label()
    layers_dict = OrderedDict()
    parents_dict = {}
    post_nms_topn = 0
    normalized_proposal = 0
    # feat_stride = 1
    if backbone_only:
        last_layer = get_last_backbone_layer(layers)
    bypass_img_input_norm = None

    is_stop = False
    for layer in layers.values():
        # print(l.name, l.type, l.params, l.parents, l.blobs.keys())
        module_name = layer.name.replace('.', '_')  # ModuleDict does not allow '.' in module name string
        if is_stop or (stop_layer is not None and layer.name == stop_layer):
            break
        if last_layer is not None and layer.name == last_layer:
            is_stop = True
        if layer.type == 'Input':
            moduledict[module_name] = gen_input(layer)
        elif layer.type == 'Const':
            moduledict[module_name] = gen_const(layer)
        elif layer.type == 'ScaleShift':
            pl = layers[layer.parents[0]]
            if pl.type in ['FullyConnected', 'ScaleShift']:
                if batch_size == 1 or not training:
                    moduledict[module_name] = gen_batchnorm(layer, pl, False)
                else:
                    moduledict[module_name] = gen_batchnorm(layer, pl, True)
            elif normalized_img_input and len(layer.parents) == 1 and layers_dict[layer.parents[0]] == 'Image_Input':
                if bypass_img_input_norm is not None:
                    print("bypass_img_input_norm is not None!!!!!!!!!!!!!!!!!!")
                bypass_img_input_norm = [layer.children, module_name, layer.parents[0].replace('.', '_')]
                continue
            else:
                moduledict[module_name] = gen_scaleshift(layer)
        elif layer.type == 'Convolution':
            if (batch_size == 1 or not training):
                moduledict[module_name] = gen_convolution(layer, False)
            else:
                moduledict[module_name] = gen_convolution(layer, True)
        elif layer.type == 'FullyConnected':
            moduledict[module_name] = gen_fc(layer)
        elif layer.type == 'Pooling':
            moduledict[module_name] = gen_pooling(layer)
        elif layer.type == 'elu':
            moduledict[module_name] = gen_elu(layer)
        elif layer.type == 'ReLU':
            moduledict[module_name] = gen_relu(layer)
        elif layer.type == 'Eltwise':
            moduledict[module_name] = gen_eltwise(layer)
        elif layer.type == 'Concat':
            moduledict[module_name] = gen_concat(layer)
        elif layer.type == 'Power':
            moduledict[module_name] = gen_power(layer)
        elif layer.type == 'Crop':
            moduledict[module_name] = gen_crop(layer)
        elif layer.type == 'Sigmoid':
            moduledict[module_name] = gen_sigmoid(layer)
        elif layer.type == 'Permute':
            moduledict[module_name] = gen_permute(layer)
        elif layer.type == 'Interp':
            moduledict[module_name] = gen_interpolate(layer, layers[layer.parents[0]], is_export)
        elif layer.type == 'Reshape':
            if len(layer.parents) > 1:
                parents_l = layers[layer.parents[1]]
                moduledict[module_name] = gen_reshape(layer, parents_l, post_nms_topn)
            else:
                moduledict[module_name] = gen_reshape(layer)
        elif layer.type == 'SoftMax':
            moduledict[module_name] = gen_softmax(layer)
        elif layer.type == 'Proposal':
            post_nms_topn = layer.params.get('post_nms_topn', 0)
            # feat_stride = int(layer.params.get('feat_stride', 1))
            normalized_proposal = int(layer.params.get('normalize', 0))
            continue
        elif layer.type == 'MVN':
            moduledict[module_name] = gen_mvn(layer)
        else:
            print('======> extra layers: ', module_name, layer.type, layer.params)
            continue
        # make network architecture information
        if layer.type == 'Input' and len(layer.out_data[0].shape) == 4:
            layers_dict[module_name] = 'Image_Input'
        else:
            layers_dict[module_name] = layer.type
        parents_dict[module_name] = list(map(lambda x: x.replace('.', '_'), list(layer.parents)))

    # remove image input normalization layer
    if normalized_img_input:
        for layer in bypass_img_input_norm[0]:
            if layer in parents_dict:
                idx = parents_dict[layer].index(bypass_img_input_norm[1])
                parents_dict[layer].insert(idx, bypass_img_input_norm[2])
                parents_dict[layer].remove(bypass_img_input_norm[1])

    return moduledict, normalized_proposal, layers_dict, parents_dict


class OMZModel(nn.Module):
    def __init__(self, moduledict, layers_info, parents_info, out_feature_names=[], **kwargs):
        super(OMZModel, self).__init__(**kwargs)
        self.model = nn.ModuleDict(list(moduledict.items()))
        self.layers_info = layers_info
        self.parents_info = parents_info
        self.featuredict = OrderedDict()
        self.softmaxdict = {}
        self.logitdict = {}
        self.out_feature_names = out_feature_names

    def forward(self, inputs, gt_label=None):
        self.featuredict.clear()
        self.softmaxdict.clear()
        if gt_label is not None:
            hidden_layer = self.model['gt_label']
            self.featuredict['gt_label'] = hidden_layer(gt_label)

        input_idx = 1
        for l_name in self.layers_info:
            hidden_layer = self.model[l_name]
            l_type = self.layers_info[l_name]
            parents = list(self.parents_info[l_name])
            if l_type == 'Proposal':
                parents.append('gt_label')
            input_size = len(parents)
            # print('Layer: ', l_name, ': ', l_type, ' <<< ', parents)
            if input_size == 1:
                input_feature = self.featuredict.get(parents[0], None)
                if l_type == 'FullyConnected' and len(input_feature.shape) > 2:
                    new_dim = [input_feature.shape[0], -1]
                    input_feature = input_feature.view(new_dim)
                feature = hidden_layer(input_feature)
            elif input_size > 1:
                input_list = []
                for input_name in parents:
                    input_feature = self.featuredict.get(input_name, None)
                    if input_feature is None:
                        print('+++> Missing input feature: ', input_name)
                    else:
                        input_list.append(input_feature)
                if len(input_list) == 1:
                    feature = hidden_layer(input_list[0])
                else:
                    if l_type == 'Proposal':  # for PVD structure
                        input_list[0] = torch.sigmoid(input_list[0])
                    feature = hidden_layer(input_list)
            elif l_type == 'Image_Input':
                feature = hidden_layer(inputs[0])
            elif l_type == 'Input':
                feature = hidden_layer(inputs[input_idx])
                input_idx += 1
            else:
                feature = hidden_layer()

            # handling output features
            if l_type == 'Proposal':
                self.featuredict['__RPN_cls_prob'] = self.featuredict[parents[0]]
                self.featuredict['__RPN_bbox_pred'] = self.featuredict[parents[1]]
                self.featuredict['__FRCNN_loss_input'] = feature
                self.featuredict['__Proposal_output'] = feature[0]
                feature = feature[0]
            elif l_type == 'ROIPooling':
                self.featuredict['__ROIPooling_output'] = feature
            elif l_type == 'PSROIPooling':
                self.featuredict['__PSROIPooling_output'] = feature
            elif l_type == 'SoftMax':
                self.logitdict[l_name] = self.featuredict[parents[0]]
                self.softmaxdict[l_name] = feature
            self.featuredict[l_name] = feature

        out_features = []
        if self.out_feature_names:
            for l_name in self.out_feature_names:
                out_features.append(self.featuredict[l_name])
        else:
            out_features.append(feature)

        return out_features

    def get_feature(self, layer_name):
        return self.featuredict.get(layer_name, None)

    def get_feature_keys(self):
        return self.featuredict.keys()

    def get_softmax_out(self, layer_name):
        return self.softmaxdict.get(layer_name, None)

    def get_softmax_out_keys(self):
        return self.softmaxdict.keys()

    def get_logit(self, layer_name):
        return self.logitdict.get(layer_name, None)

    def get_logit_keys(self):
        return self.logitdict.keys()
