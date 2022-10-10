# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, List, Optional, Union

import torch
from mpa.utils.logger import get_logger

from ..graph import Graph
from ..graph.utils import handle_merging_into_batchnorm, handle_paired_batchnorm
from ..ops import OPS
from ..utils import load_ov_model, normalize_name


logger = get_logger()


CONNECTION_SEPARATOR = "||"


class OVModel(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        weight_path: Optional[str] = None,
        inputs: Union[str, List[str]] = [],
        outputs: Union[str, List[str]] = [],
        features_to_keep: Optional[List] = None,
        remove_normalize: bool = False,
        merge_bn: bool = True,
        paired_bn: bool = True,
        init_weight: Union[bool, Callable] = False,
        verify_shape: bool = True,
    ):
        super().__init__()
        self._model_path = model_path
        self._weight_path = weight_path
        self._remove_normalize = remove_normalize
        self._features_to_keep = features_to_keep
        self._merge_bn = merge_bn
        self._paired_bn = paired_bn
        self._init_weight = init_weight
        self._verify_shape = verify_shape

        self._inputs = []
        self._outputs = []
        self._feature_dict = OrderedDict()

        # build graph
        graph = self.build_graph(model_path, weight_path)
        self._graph = graph
        if remove_normalize:
            graph.remove_normalize_nodes()

        # handle inputs
        if inputs:
            inputs = inputs if isinstance(inputs, list) else [inputs]
            assert all(
                isinstance(i, str) for i in inputs
            ), f"input must be string but {inputs} is given"
            inputs = self.build_custom_inputs(graph, deepcopy(inputs))
        else:
            inputs = [node.name for node in graph.get_nodes_by_types(["Parameter"])]
        self._inputs = inputs

        # handle outputs
        if outputs:
            outputs = outputs if isinstance(outputs, list) else [outputs]
            assert all(
                isinstance(i, str) for i in outputs
            ), f"input must be string but {outputs} is given"
            outputs = self.build_custom_outputs(graph, deepcopy(outputs))
        else:
            outputs = [node.name for node in graph.get_nodes_by_types(["Result"])]
        self._outputs = outputs

        # clean up graph
        self.clean_up(graph, inputs, outputs)

        if merge_bn:
            handle_merging_into_batchnorm(graph)
        if paired_bn:
            handle_paired_batchnorm(graph, replace=True)

        # clean up graph
        self.clean_up(graph, inputs, outputs)

        # build torch module
        self.model = self.build_torch_module(graph)

        if init_weight:
            if not isinstance(init_weight, Callable):

                def init_weight(m):
                    from ..ops.op import Operation

                    if not isinstance(m, Operation):
                        return
                    if (
                        not m.name.endswith("bn/gamma")
                        and not m.name.endswith("bn/beta")
                        and not m.name.endswith("bn/running_mean")
                        and not m.name.endswith("bn/running_variance")
                    ):
                        if getattr(m, "data", None) is not None:
                            if isinstance(m.data, torch.nn.parameter.Parameter):
                                try:
                                    torch.nn.init.xavier_normal_(m.data)
                                    logger.info(f"Initialize parameter {m.name}")
                                except Exception:
                                    logger.info(
                                        f"Failed to initialize parameter {m.name}"
                                    )

            self.model.apply(init_weight)
        for node in self._graph.get_nodes_by_types(["Parameter"]):
            node.attrs.verify_shape = verify_shape

        input_shapes = {}
        output_shapes = {}
        for node in self._graph.get_nodes_by_types(["Parameter", "Result"]):
            if node.name in self._inputs:
                input_shapes[node.name] = node.shape[0]
            elif node.name in self._outputs:
                output_shapes[node.name] = node.shape[0]
        self._input_shapes = OrderedDict()
        self._output_shapes = OrderedDict()
        for input in self._inputs:
            self._input_shapes[input] = input_shapes[input]
        for output in self._outputs:
            self._output_shapes[output] = output_shapes[output]

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def features(self):
        return self._feature_dict

    @property
    def input_shapes(self):
        return self._input_shapes

    @property
    def output_shapes(self):
        return self._output_shapes

    @staticmethod
    def build_graph(model_path, weight_path=None):
        # TODO: reshape decompose ir graph
        ov_model = load_ov_model(model_path, weight_path, False)
        graph = Graph.from_ov(ov_model)
        graph.clean_up()
        return graph

    @staticmethod
    def build_custom_outputs(graph, outputs):
        cls_result = OPS.get_by_type_version("Result", 0)
        node_dict = OrderedDict((i.name, i) for i in graph.topological_sort())

        if not isinstance(outputs, list):
            outputs = [outputs]

        edges_to_add = []
        nodes_to_remove = []
        for i, output in enumerate(outputs):
            output = normalize_name(output)
            output = output.split(CONNECTION_SEPARATOR)
            explicit_tgt = False

            if len(output) == 1:
                src = output[0]
                tgt = None
            elif len(output) == 2:
                src, tgt = output
                explicit_tgt = True
            else:
                raise ValueError()

            src = node_dict[src]
            if src.type == "Result":
                continue

            if explicit_tgt:
                tgt = node_dict[tgt]
            else:
                tgt = list(graph.successors(src))[0]

            output_result = f"{src.name}/result_{i}"
            outputs[i] = output_result

            for successor in graph.successors(src):
                if tgt == successor:
                    edges_attrs = graph.get_edge_data(src, successor)
                    assert len(edges_attrs) == 1
                    for edge_attrs in edges_attrs:
                        edge_attrs["in_port"] = 0

                    output_result = cls_result(output_result, shape=src.shape)
                    for edge_attrs in edges_attrs:
                        edges_to_add.append(
                            {"node_from": src, "node_to": output_result, **edge_attrs}
                        )
                if explicit_tgt and tgt != successor:
                    continue
                nodes_to_remove.append(successor)

        for node in set(nodes_to_remove):
            graph.remove_node(node)
        for edge in edges_to_add:
            graph.add_edge(**edge)
        return outputs

    @staticmethod
    def build_custom_inputs(graph, inputs: Union[str, List[str]]):
        cls_param = OPS.get_by_type_version("Parameter", 0)
        node_dict = OrderedDict((i.name, i) for i in graph.topological_sort())

        if not isinstance(inputs, list):
            inputs = [inputs]

        edges_to_add = []
        nodes_to_remove = []
        paired_nodes_to_remove = []
        for i, input in enumerate(inputs):
            input = normalize_name(input)
            input = input.split(CONNECTION_SEPARATOR)
            explicit_src = False

            if len(input) == 1:
                src = None
                tgt = input[0]
            elif len(input) == 2:
                src, tgt = input
                explicit_src = True
            else:
                raise ValueError()

            tgt = node_dict[tgt]
            if tgt.type == "Parameter":
                continue

            if explicit_src:
                src = node_dict[src]
            else:
                src = list(graph.predecessors(tgt))[0]

            input_parameter = f"{tgt.name}/parameter_{i}"
            inputs[i] = input_parameter

            for predecessor in graph.predecessors(tgt):
                if src == predecessor:
                    edges_attrs = graph.get_edge_data(predecessor, tgt)
                    assert len(edges_attrs) == 1
                    for edge_attrs in edges_attrs:
                        edge_attrs["out_port"] = 0

                    # TODO: here, we force the batch dim to be dynamic
                    # it is assumed to be dim 0
                    new_shape = []
                    for shape in predecessor.shape:
                        new_shape.append(
                            [-1 if j == 0 else k for j, k in enumerate(shape)]
                        )
                    new_shape = tuple(tuple(shape) for shape in new_shape)
                    input_parameter = cls_param(input_parameter, shape=new_shape)
                    for edge_attrs in edges_attrs:
                        edges_to_add.append(
                            {"node_from": input_parameter, "node_to": tgt, **edge_attrs}
                        )
                if (
                    explicit_src and src != predecessor
                ) or predecessor.type == "Constant":
                    continue
                nodes_to_remove.append(predecessor)
                paired_nodes_to_remove.append(tgt)

        # handle duplicated predecessors
        if len(set(nodes_to_remove)) != len(nodes_to_remove):
            done = [False for _ in nodes_to_remove]
            input_pop_indices = []
            for tgt_idx, tgt in enumerate(nodes_to_remove):
                if done[tgt_idx]:
                    continue
                indices = [
                    i + tgt_idx + 1
                    for i, x in enumerate(nodes_to_remove[tgt_idx + 1:])
                    if x == tgt
                ]
                done[tgt_idx] = True

                need_to_update = []
                for edge_idx, edge in enumerate(edges_to_add):
                    if edge["node_to"] == paired_nodes_to_remove[tgt_idx]:
                        tgt_edge = edge
                    else:
                        for idx in indices:
                            if edge["node_to"] == paired_nodes_to_remove[idx]:
                                need_to_update.append((edge_idx, idx))
                for i, (edge_idx, idx) in enumerate(need_to_update[::-1]):
                    edges_to_add[edge_idx]["node_from"] = tgt_edge["node_from"]
                    done[idx] = True
                    input_pop_indices.append(edge_idx)
            for idx in sorted(input_pop_indices, reverse=True):
                inputs.pop(idx)

        for node in set(nodes_to_remove):
            graph.remove_node(node)
        for edge in edges_to_add:
            graph.add_edge(**edge)
        return inputs

    @staticmethod
    def clean_up(graph, inputs=[], outputs=[]):
        nodes = list(graph.topological_sort())
        nodes_to_keep = []
        for node in nodes:
            if node.name in inputs or node.name in outputs:
                nodes_to_keep.append(node)

        def get_nodes_without_successors(graph, ignores=[]):
            outputs = []
            for node in reversed(list(graph.topological_sort())):
                if not list(graph.successors(node)) and node not in ignores:
                    outputs.append(node)
            return outputs

        nodes = get_nodes_without_successors(graph, nodes_to_keep)
        while nodes:
            graph.remove_nodes_from(nodes)
            nodes = get_nodes_without_successors(graph, nodes_to_keep)

        graph.clean_up(nodes_to_keep)

    @staticmethod
    def build_torch_module(graph):
        node_dict = OrderedDict((i.name, i) for i in graph.topological_sort())
        return torch.nn.ModuleDict(list(node_dict.items()))

    def _build_forward_inputs(self, *args, **kwargs):
        inputs = {}
        if args:
            for key, arg in zip(self._inputs, args):
                inputs[key] = arg
        if kwargs:
            for key, arg in kwargs.items():
                if key in inputs:
                    raise ValueError
                inputs[key] = arg
        return inputs

    def forward(self, *args, **kwargs):
        self._feature_dict.clear()
        inputs = self._build_forward_inputs(*args, **kwargs)

        done = {}
        for node_name, node in self.model.items():
            done[node_name] = {
                node.name: False for node in self._graph.successors(node)
            }

        for node_name, node in self.model.items():
            predecessors_with_edge = list(
                self._graph.predecessors(node, with_edge_data=True)
            )
            if not predecessors_with_edge:
                if node.type == "Parameter":
                    self._feature_dict[node_name] = node(inputs[node_name])
                elif node.type == "Constant":
                    self._feature_dict[node_name] = node()
                else:
                    raise ValueError(
                        f"Broken graph. Node {node_name} is a type of {node.type} "
                        "but it has no in edges."
                    )
            else:
                input_nodes, edges = list(map(list, zip(*predecessors_with_edge)))
                input_node_names = [input_node.name for input_node in input_nodes]

                input_features = [
                    edge["in_port"] for edges_ in edges for edge in edges_
                ]
                assert len(input_features) == len(set(input_features))
                input_features = [None for _ in input_features]
                for idx, input_node_name in enumerate(input_node_names):
                    if (
                        self._features_to_keep is not None
                        and input_node_name in self._features_to_keep
                    ):
                        input_feature = self._feature_dict.get(input_node_name)
                    else:
                        input_feature = self._feature_dict.pop(input_node_name)
                        done[input_node_name][node_name] = True
                        if not all(done[input_node_name].values()):
                            self._feature_dict[input_node_name] = input_feature

                    if isinstance(input_feature, tuple):
                        for edges_ in edges[idx]:
                            input_features[edges_["in_port"]] = input_feature[
                                edges_["out_port"]
                            ]
                    else:
                        for edges_ in edges[idx]:
                            input_features[edges_["in_port"]] = input_feature
                assert all(
                    input_feature is not None for input_feature in input_features
                )
                self._feature_dict[node_name] = node(*input_features)

        outputs = OrderedDict()
        for output_name in self._outputs:
            outputs[output_name] = self._feature_dict[output_name]

        return outputs
