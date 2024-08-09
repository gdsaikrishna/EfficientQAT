#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from mmdnn.conversion.onnx.graph_pruning import SubgraphEvaluation
from mmdnn.conversion.onnx.constant_folding import ConstantFoldingImpl
import torch
import csv
import os
import re
import traceback
from pprint import pprint as pp
import torch.jit
import torch.autograd
import torch.serialization
import torch.onnx.utils
from collections import OrderedDict
from mmdnn.conversion.common.utils import layerLikeObj
import contextlib
from torch.jit import _unique_state_dict
from collections import defaultdict
import onnx.helper as OH
import onnx.numpy_helper as NH
import onnxruntime
import onnx
from onnx import TensorProto
import numpy as np
import itertools
import io
from random import randint
import torch.nn.functional as F
from copy import deepcopy
from utils.dict2object import ObjectLike
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import mmdnn.conversion.onnx.onnx_graph_utils as netx
from dv_logging import get_logger

logger = get_logger(__name__, 'DVGRPH')


def _normalize_attrs(kv):
  """converts any torch specific datastructure
    into standard containers / types.
    use this on node attributes
    """
  for key in kv.keys():
    value = kv[key]
    if type(value) == torch.Tensor:
      kv[key] = value.numpy()
  return kv


ATTR_MAP = {onnx.AttributeProto.AttributeType.INT: 'i', onnx.AttributeProto.AttributeType.FLOAT: 'f', onnx.AttributeProto.AttributeType.STRING: 's', onnx.AttributeProto.AttributeType.TENSOR: 't', onnx.AttributeProto.AttributeType.GRAPH: 'g', onnx.AttributeProto.AttributeType.SPARSE_TENSOR: 'sparse_tensor', onnx.AttributeProto.AttributeType.FLOATS: 'floats', onnx.AttributeProto.AttributeType.INTS: 'ints', onnx.AttributeProto.AttributeType.STRINGS: 'strings', onnx.AttributeProto.AttributeType.TENSORS: 'tensors', onnx.AttributeProto.AttributeType.GRAPHS: 'graphs', onnx.AttributeProto.AttributeType.SPARSE_TENSORS: 'sparse_tensors'}


class OnnxGraphNode(GraphNode):

  def __init__(self, layer, id):

    def _get_attr(attr_proto):
      ptype = attr_proto.type
      return getattr(attr_proto, ATTR_MAP[ptype])

    self._layer = layer
    self._name = layer.name
    try:
      if layer.HasField('op_type'):
        self._kind = layer.op_type
    except ValueError:
      self._kind = 'Input'
    self.shape = None
    self.initializers = []
    self.id = id
    super(OnnxGraphNode, self).__init__(layer)
    self.attrs = defaultdict(list)
    self.weights_name = ''
    if self._kind not in ['Input']:
      for k in layer.attribute:
        self.attrs[k.name] = _get_attr(k)

  def set_node_type(self, type):
    self._kind = type

  @staticmethod
  def mk_node_name(name):
    # Scopes created in a nested scope may have initial characters
    # that are illegal as the initial character of an op name
    # (viz. '-', '\', '/', and '_').
    name = name.replace('-','_')\
               .replace('\\','_')\
               .replace('/','_')\
               .replace('_','_')\
               .replace('[','_')\
               .replace(']','_')\
               .replace(':', '_')
    return '{}'.format(name)

  @property
  def name(self):
    name = self._name
    return OnnxGraphNode.mk_node_name(name)

  # TODO Unused code
  # @staticmethod
  # def get_name_and_id_for(layer):
  #   scope_name = layer.scopeName()
  #   node_id = re.search(r"[\d]+", layer.__str__())
  #   node_id = node_id.group(0)
  #   if not scope_name:
  #     scope_name = '{}_generated'.format(node_id)
  #   node_name = scope_name + node_id
  #   node_name = node_name.replace('-','_') \
  #                        .replace('\\','_') \
  #                        .replace('/','_') \
  #                        .replace('_','_') \
  #                        .replace('[','_') \
  #                        .replace(']','_')
  #   obj_arg = dict(node_id=node_id, node_name=node_name)
  #   return ObjectLike(obj_arg)

  @property
  def is_deterministic(self):
    pass

  @property
  def type(self):
    return self._kind

  @property
  def pytorch_layer(self):
    return self.layer

  def __str__(self):
    return 'ONNXGraphNode({}, {})'.format(self.name, self.type)

  def __repr__(self):
    return self.__str__()

  def get_attr(self, name, default_value=None):
    if self.attrs:
      if name in self.attrs.keys():
        return self.attrs.get(name)
      else:
        return default_value
    else:
      return default_value


class OnnxGraph(Graph):

  def __init__(self, model, all_inference_model, input_node, output_node, onnx_backend="onnxruntime", images=None, data_format=None, dformat=None):
    # sanity check.
    super(OnnxGraph, self).__init__(model)
    self.model = model  #layerLikeObj(dict(state_dict=lambda **kwargs: self.model))
    self.all_inference_model = all_inference_model
    self.state_dict = {}
    self.params_dict = dict()
    self.shape_dict = dict()
    self.output_node = output_node
    self.input_node = input_node
    self._onnx_layer_type_map = defaultdict(list)
    self.layer_args_order_map = defaultdict(dict)
    self.onnx_backend = onnx_backend
    self.images = images
    self.data_format = data_format
    self.dformat = dformat

  @staticmethod
  def get_node_id(node):
    node_id = re.search(r"[\d]+", node.__str__())
    return node_id.group(0)

  def __check_outbound_type(self, nodelist, dtype):
    items = defaultdict(list)
    children_dtype = True
    for node in nodelist:
      node = self.layer_map[node]
      for edge in node.out_edges:
        out_edge = self.layer_map[edge]
        if (out_edge.type == dtype):
          items[node].append(edge)
        children_dtype = children_dtype and out_edge.type == dtype
    return items, children_dtype

  def rebuild(self):
    logger.info('norebuild')

  def normalize_layer_keys(self, _dict):
    _ret = dict()
    for k, v in _dict.items():
      _ret[OnnxGraphNode.mk_node_name(k)] = v
    return _ret

  def build(self, shape, output_dir):
    const_users = open('./constant.txt', 'w')
    reduced_nodes = open('./pruned_subgraphs.txt', 'w')
    onnx_model = onnx.load_model(self.model)
    # In case of multiple input names are needed to map to the shapes.
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = set(input_all) - set(input_initializer)
    if len(net_feed_input) > 1 and len(net_feed_input) != len(self.input_node):
      raise AssertionError("For multi-input model, provide correct input names")
    init = [n.name for n in onnx_model.graph.initializer]
    inputs = [n.name for n in onnx_model.graph.input if n.name not in init]
    onnx.shape_inference.infer_shapes_path(self.model)
    onnx_model = onnx.load(self.model)
    if len(onnx_model.graph.input) == 1:
      self.input_node = inputs
    nodes = onnx_model.graph.node
    output_nodes = set(map(lambda it: it.name, onnx_model.graph.output))
    logger.info("num nodes in original model is %d, output_nodes: %d", len(nodes), len(output_nodes))
    input_initializers = {}
    for i in self.input_node:
      input_initializers[i] = dict()
    self.state_dict = {}
    tensor_mapping = {}
    shape_mapping = {}
    type_mapping = {}
    for entry in onnx_model.graph.initializer:
      name = OnnxGraphNode.mk_node_name(entry.name)
      npa = NH.to_array(entry)
      self.state_dict[name] = npa
      shape_mapping[name] = list(npa.shape)
    for entry in onnx_model.graph.node:
      if entry.op_type == 'Constant':
        self.state_dict[entry.output[0]] = NH.to_array(entry.attribute[0].t)
    for entry in (list(onnx_model.graph.value_info) + list(onnx_model.graph.output)):
      name = OnnxGraphNode.mk_node_name(entry.name)
      type_mapping[name] = entry.type.tensor_type.elem_type
      shape_mapping[name] =\
          list(map(lambda x: x.dim_value,
                   entry.type.tensor_type.shape.dim))
    self.shape_dict = netx.OnnxInference().get_output_shapes(self.model, self.all_inference_model, self.input_node, shape, self.onnx_backend, self.images, self.data_format, self.dformat)
    G = netx.nx_graph_object(onnx_model)
    r'''make sure input and output nodes are in graph. 
           else ignore Subgraph.'''
    if self.output_node == None:
      self.output_node = 'None'
    subgraph = list(itertools.product(self.input_node, self.output_node))
    for inp, out in subgraph:
      #            if inp not in G.node or out not in G.node:
      if inp not in G.nodes or out not in G.nodes:
        logger.info('Ignoring Subgraph eval. Invalid input or output node names')
        self.input_node = None
        self.output_node = None
        break
    G = netx.onnx_subGraph_match_replace_Centernet_IR1_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_PytorchOnnxSoftmax_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_PytorchOnnxSoftmax_PSP_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_ReduceSum_To_ELTS3_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_ReduceSum_To_ELTS2_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_Centernet_IR4_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_Swish_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L1NormMatcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_zt(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_AC(G, onnx_model, self.shape_dict, self.state_dict)
    G = netx.onnx_subGraph_match_replace_L2NormScale_Matcher(G, onnx_model, self.shape_dict, self.state_dict)

    # G = netx.onnx_subGraph_match_replace_concat_Matcher(G, onnx_model, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_yolov8_scatterND_Matcher(G, onnx_model, self.state_dict, self.shape_dict)

    G = netx.onnx_subGraph_match_replace_Mish_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_Yolo4_Expand_To_Upsample_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_simba(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_simba_1(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_clip(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_Advertima_IR(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_1(G, onnx_model)
    ## don't change order
    G = netx.onnx_subGraph_match_replace_Convnext_simplify(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_LayerNormMatcher_1(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_LayerNormMatcher_2(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_LayerNormMatcher_4(G, onnx_model)
    # simplifier needs to be above gelu
    G = netx.onnx_subGraph_match_replace_GeluLayer_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_GeluLayer2_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_LayerNormP1_Matcher(G, onnx_model, self.state_dict)
    G = netx.onnx_subGraph_match_replace_LayerNormP2_Matcher(G, onnx_model, self.state_dict)
    G = netx.onnx_subGraph_match_replace_SFRoiLayer_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_SFRoi2Layer_Matcher(G, onnx_model)
    # G = netx.onnx_subGraph_match_replace_SFRoiMaxLayer_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_LayerNormMatcher_3(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_RoiAlignMax_Matcher(G, onnx_model, self.state_dict)
    ##
    G = netx.onnx_subGraph_match_replace_FocusLayer_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_FocusLayer2_Matcher(G, onnx_model)
    # G = netx.onnx_subGraph_match_replace_WeedOut_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_RepeatInterleave_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_DensePose_upsample_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_ConcatUpsample_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher_2(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_GroupNormMatcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_GroupNorm1Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Flynn_d_tail_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_AliaswithNameRoi_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_GroupnormCast_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Unet_1(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Unet_2(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Unet_3(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Unet_4(G, onnx_model, self.state_dict, self.shape_dict)
    #G = netx.onnx_subGraph_math_replace_RMSNormMatcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_RMSNormMatcher2(G, onnx_model, self.state_dict, self.shape_dict)
    # G = netx.onnx_subGraph_match_replace_UnsqueezeAfterSinCos_Matcher_Matcher(G, onnx_model, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_llama_unsqueeze_gather_Matcher(G, onnx_model, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Rope_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_RopeTransposeMul_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_RopeMul_Matcher(G, onnx_model)
    # G = netx.onnx_subGraph_match_replace_GCD_Matcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_IBN(G, onnx_model, self.state_dict)
    cur_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(cur_path, 'utils/patterns/priorbox')
    for priorbox in os.listdir(dir_path):
      filepath = os.path.join(dir_path, priorbox)
      G = netx.onnx_subGraph_match_replace_PriorBoxDotMatcher(G, onnx_model, filepath, self.shape_dict, self.state_dict)
    dir_path = os.path.join(cur_path, 'utils/patterns/detectionbox')
    for detectionbox in os.listdir(dir_path):
      filepath = os.path.join(dir_path, detectionbox)
      G = netx.onnx_subGraph_match_replace_DetectionBoxMatcher(G, onnx_model, filepath, self.shape_dict)
    dir_path = os.path.join(cur_path, 'utils/patterns/bboxtransform_clip')
    for detectionbox in os.listdir(dir_path):
      filepath = os.path.join(dir_path, detectionbox)
      G = netx.onnx_subGraph_match_replace_BBoxTransformDotMatcher(G, onnx_model, filepath, self.shape_dict, self.state_dict)
    dir_path = os.path.join(cur_path, 'utils/patterns/detectron2')
    for detectionbox in os.listdir(dir_path):
      filepath = os.path.join(dir_path, detectionbox)
      G = netx.onnx_subGraph_match_replace_BBoxTransformDotMatcher(G, onnx_model, filepath, self.shape_dict, self.state_dict)
    dir_path = os.path.join(cur_path, 'utils/patterns/bboxtransform_static')
    for detectionbox in os.listdir(dir_path):
      filepath = os.path.join(dir_path, detectionbox)
      G = netx.onnx_subGraph_match_replace_BBoxTransformDotStaticMatcher(G, onnx_model, filepath, self.shape_dict, self.state_dict)
    G = netx.onnx_subGraph_match_replace_L2NormMatcher(G, onnx_model)
    G = netx.onnx_subGraph_match_replace_Mul2_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_FullyConnected(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_MatMul(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_Tile_Mul_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_NullPattern_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstCast_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstPow_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstGather_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstReshape_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstUnsqueeze_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstDiv_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    G = netx.onnx_subGraph_match_replace_ConstAdd_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    # G = netx.onnx_subGraph_match_replace_ZeroTensorAdd_Matcher(G, onnx_model, self.state_dict, self.shape_dict)
    # G = netx.onnx_subGraph_match_remove_CastReshape1(G, onnx_model)
    # G = netx.onnx_subGraph_match_remove_CastReshape2(G, onnx_model)
    # G = netx.onnx_subGraph_match_replace_ReshapeUnsqueeze(G, onnx_model, self.shape_dict)
    ''' Experimental : netx.NXtoGraph().nx_to_onnx_graph_object(G, onnx_model, shape)
            Dumps nx to onnx graph, for inference. [May crash]
        '''
    self.shape_dict = self.normalize_layer_keys(self.shape_dict)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = [i for i in input_all if i not in input_initializer]
    # net_feed_input = list(set(input_all)  - set(input_initializer))
    dummy_node_map = dict()
    for idx, node in enumerate(onnx_model.graph.input):
      if node.name in net_feed_input:
        self.layer_map[OnnxGraphNode.mk_node_name(node.name)] = OnnxGraphNode(node, str(idx))
        self.layer_name_map[OnnxGraphNode.mk_node_name(node.name)] = node.name
        tensor_mapping[OnnxGraphNode.mk_node_name(node.name)] = node

    node_names = []
    topological_sorted_graph = netx.topological_sort(G)
    for node_name in topological_sorted_graph:
      G_temp = G.nodes[node_name]
      if G_temp.get('type', None) is None:
        continue
      if G_temp['type'] == 'input' or G_temp['type'] == 'tensor':
        continue
      node_names.append(node_name)
    for idx, node_name in enumerate(node_names):
      _idx = idx + len(onnx_model.graph.input)
      G_temp = G.nodes[node_name]
      if G_temp.get('type') == 'dv_remove':
        logger.info("Removing the dummy node")
        dummy_node_map[G_temp.get('data').output[0]] = G_temp
        continue
      node = G_temp.get('data', None)
      curr_node = OnnxGraphNode(node, str(_idx))
      node_name = curr_node.name
      self._onnx_layer_type_map[curr_node.type].append(node_name)
      for output in node.output:
        output = OnnxGraphNode.mk_node_name(output)
        tensor_mapping[output] = curr_node
      self.layer_map[curr_node.name] = curr_node
      self.layer_name_map[curr_node.name] = curr_node.name
      node_inputs = node.input
      count = 0
      for _input_idx, _node_input in enumerate(node_inputs):
        # if _node_input in dummy_node_map:
        #   if len(dummy_node_map[_node_input].get('data').input) == 0:
        #     logger.info("Skipping the dv_remove input")
        #     continue
        #   _node_input = dummy_node_map[_node_input].get('data').input[0]
        node_input = OnnxGraphNode.mk_node_name(_node_input)
        if self.state_dict.get(node_input, None) is not None:
          count += 1
          if count == 1:
            curr_node.weights_name = node_input
          self.state_dict[curr_node.weights_name + '_' + str(count)] = self.state_dict[node_input]
          curr_node.initializers.append(node_input)
          continue
        self.layer_args_order_map[curr_node.name][node_input] = _input_idx
        if _idx > 0:
          if node_input in tensor_mapping:
            logger.debug('{%s} -> {%s}', tensor_mapping[node_input].name, node_name)
            self._make_connection(tensor_mapping[node_input].name, node_name)
            # self.layer_map[node_name.split(':')[0]].in_blob_edges.append(node_input)
            self.layer_map[node_name.split(':')[0]].in_blob_edges.append(tensor_mapping[node_input].name)
      output_name = 'None'
      if len(node.output) != 0:
        output_name = OnnxGraphNode.mk_node_name(node.output[0])
      else:
        logger.error("outputs are not found")
      if output_name in self.shape_dict:
        shape_data = self.shape_dict[output_name]
        # ADI: TODO:
        # review risks in hardcoding types to 1
        if output_name in type_mapping:
          type_data = type_mapping[output_name]
        else:
          type_data = 1
        self.shape_dict[curr_node.name] = shape_data
        curr_node.shape = (shape_data, type_data)
    for idx, inp in enumerate(net_feed_input):
      self.shape_dict[inp] = shape[idx]
    try:
      node_variety = defaultdict(list)
      for key, value in self.layer_map.items():
        node_variety[value.type].append(value.name)
    except Exception as e:
      traceback.print_exc()

    def get_non_constant_parent(layer):
      for in_edge in self.layer_map[layer].in_edges:
        if self.layer_map[in_edge].type != 'Constant':
          return in_edge
      return None

    def patch_topsort_placement(topsort, nodes_to_place):
      input_layers = _get_input_layers()
      for node in nodes_to_place:
        consumer = self.layer_map[node].out_edges[0]
        child_index = topsort.index(consumer)
        topsort.insert(child_index, node)
        self.layer_map[input_layers[0]].out_edges.append(node)
        self.layer_map[node].in_edges.append(input_layers[0])

    def _traverse2(graph):
      for key, value in self.layer_map.items():
        # for out_edge in value.layer.output:
        for out_edge in value.out_edges:
          graph.add_edge(key, out_edge)

    def _dump_graph():
      graph = nx.DiGraph()
      visited = set()
      # input_layers = _get_input_layers()
      # constant_layers = _get_constant_layers()
      # for layer in constant_layers:
      #     self.layer_map[layer].in_edges.append(input_layers[0])
      #     self.layer_map[input_layers[0]].out_edges.append(layer)
      #     _traverse2(graph)
      # p
      _traverse2(graph)
      write_dot(graph, "pruned.dot")

      def _helper(it):
        node = self.layer_map[it]
        if len(node.out_edges) > 0:
          return node.out_edges[0]
        else:
          return it

      topsort = nx.lexicographical_topological_sort(graph, key=_helper)
      topsort_final = []
      input_nodes = []
      output_nodes = []
      constant_nodes = []
      for item in topsort:
        it = self.layer_map[item]
        if it.type != 'Constant' and len(it.in_edges) == 0:
          input_nodes.append(item)
        elif it.type != 'Constant' and len(it.out_edges) == 0:
          output_nodes.append(item)
        else:
          if it.type == 'Constant' and len(it.in_edges) == 0:
            constant_nodes.append(item)
          else:
            topsort_final.append(item)
      # patch_topsort_placement(topsort_final, constant_nodes)
      return topsort_final, input_nodes, output_nodes

    def _get_constant_layers():
      constant_layers = []
      for key, value in self.layer_map.items():
        if len(value.in_edges) == 0 and value.type == 'Constant':
          constant_layers.append(key)
      return constant_layers

    def _get_input_layers():
      input_layers = []
      for key, value in self.layer_map.items():
        if len(value.in_edges) == 0 and value.type != 'Constant':
          input_layers.append(key)
      return input_layers

    def _check_unreachable_nodes(visited, curr_node, terminals):
      visited.add(curr_node)
      children = self.layer_map[curr_node].out_edges
      if len(children) == 0:
        terminals.add(curr_node)
        return visited
      for child in children:
        _check_unreachable_nodes(visited, child, terminals)
      return visited

    def rebuild_layer_map(topsort):
      layer_map = OrderedDict()
      layer_name_map = OrderedDict()
      for key in topsort:
        layer_map[key] = self.layer_map[key]
        layer_name_map[key] = key
      return layer_map, layer_name_map

    const_users.close()
    reduced_nodes.close()
    with open('final_graph_layers.txt', 'w') as f:
      f.write(self.layer_map.__str__())
    logger.info("DONE BUILDING")
    visited = set()
    terminals = set()
    super(OnnxGraph, self).build()
    topological_sort, input_nodes, output_nodes = _dump_graph()
    logger.debug('topsort : [%s],\n input_nodes : [%s]', topological_sort, input_nodes)
    self.topological_sort = input_nodes + topological_sort + output_nodes
    self.input_layers = input_nodes
    self.output_layers = output_nodes
    new_layer_map, new_layer_name_map = rebuild_layer_map(self.topological_sort)
    self.layer_map = new_layer_map
    self.layer_name_map = new_layer_name_map
