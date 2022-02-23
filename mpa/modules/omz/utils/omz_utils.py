import os
import sys

from collections import OrderedDict


def get_net(model: str):
    try:
        from openvino import inference_engine as ie  # noqa: F401
        from openvino.inference_engine import IECore  # noqa: F401
    except Exception as e:
        exception_type = type(e).__name__
        print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
        sys.exit(1)

    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    iec = IECore()
    net = iec.read_network(model=model_xml, weights=model_bin)
    return net


def get_layers_list(all_layers: dict, inputs: dict, outputs: list, layers: str):
    if layers is not None and layers != 'None':
        if layers == 'all':
            return all_layers
        else:
            user_layers = [layer.strip() for layer in layers.split(',')]
            layers_to_check = []
            for user_layer in user_layers:
                if user_layer not in all_layers:
                    raise Exception("Layer {} doesn't exist in the model".format(user_layer))
                if user_layer in inputs:
                    raise Exception("Layer {} is input layer. Can not proceed".format(user_layer))
                layers_to_check.append(user_layer)
            return layers_to_check
    else:
        return outputs


def get_graph_info_from_openvino(openvino_layers):
    layers_dict = OrderedDict()
    parents_dict = {}
    for layer in openvino_layers.values():
        if layer.type == 'Input' and len(layer.out_data[0].shape) == 4:
            layers_dict[layer.name] = 'Image_Input'
        else:
            layers_dict[layer.name] = layer.type
        parents_dict[layer.name] = list(layer.parents)

    return layers_dict, parents_dict


def find_root(openvino_layers, leaf_layer, root_layer_name=None, root_id=-1):
    parent_layers = leaf_layer.parents
    print('***', root_layer_name, ' : ', root_id)
    for pl_name in parent_layers:
        pl = openvino_layers[pl_name]
        if pl.type in ['Input', 'Const']:
            continue  # im_info
        pid = list(openvino_layers.keys()).index(pl_name)
        print('...', pl_name, ' : ', pid)
        if len(pl.out_data[0].shape) != 4:
            return find_root(openvino_layers, pl, root_layer_name, root_id)
        elif root_id == -1:
            root_id = pid
            root_layer_name = pl.name
        elif root_id > pid:
            leaf_layer = openvino_layers[root_layer_name]
            return find_root(openvino_layers, leaf_layer, pl_name, pid)
        elif root_id < pid:
            return find_root(openvino_layers, pl, root_layer_name, root_id)
        else:
            break

    return root_layer_name, root_id


def get_last_backbone_layer(openvino_layers):
    for layer in openvino_layers.values():
        if layer.type == 'Softmax':  # classification case
            last_backbone_layer, _ = find_root(openvino_layers, layer, None, -1)
            break
        elif layer.type == 'Proposal':  # faster-rcnn case
            last_backbone_layer, _ = find_root(openvino_layers, layer, None, -1)
            break
        else:  # recognition case
            last_backbone_layer = layer.name

    return last_backbone_layer
