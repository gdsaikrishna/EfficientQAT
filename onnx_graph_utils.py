import networkx as nx
from networkx.algorithms import isomorphism
import sys
import os
import itertools
import numpy as np
import pickle
import onnx
from onnx import helper
import onnxruntime
import utils.io_utils as io_utils
import onnx.numpy_helper as nh
from collections import namedtuple
from onnx import AttributeProto, TensorProto, GraphProto
from networkx.drawing.nx_pydot import write_dot

sys.path.append(os.getcwd() + '/' + os.path.dirname(__file__) + '/../../caffe/python')
sys.path.append(os.getcwd() + '/' + os.path.dirname(__file__) + '/../')
from dv_logging import get_logger
from copy import deepcopy
from typing import List, Any

logger = get_logger(__name__, 'DVONNXUTIL')
_sess_options = onnxruntime.SessionOptions()
_sess_options.inter_op_num_threads = 1
_sess_options.intra_op_num_threads = 1


def write_dot_colon_wrapper(graph, path):
  graph_temp = deepcopy(graph)
  for node in graph_temp.nodes:
    # print("serializing node:", node)
    graph_temp.nodes[node]['data'] = 'data'
    graph_temp.nodes[node]['type'] = str(graph_temp.nodes[node]['type']).replace(":", "|")
  write_dot(graph_temp, path)


def add_value_info_for_constants(model):
  """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
  # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
  if model.ir_version < 4:
    return

  def add_const_value_infos_to_graph(graph):
    inputs = {i.name for i in graph.input}
    existing_info = {vi.name: vi for vi in graph.value_info}
    for init in graph.initializer:
      # Check it really is a constant, not an input
      if init.name in inputs:
        continue
      # The details we want to add
      elem_type = init.data_type
      shape = init.dims
      # Get existing or create new value info for this constant
      vi = existing_info.get(init.name)
      if vi is None:
        vi = graph.value_info.add()
        vi.name = init.name
      # Even though it would be weird, we will not overwrite info even if it doesn't match
      tt = vi.type.tensor_type
      if 'TensorProto' in str(type(tt.elem_type)):
        tt.elem_type = elem_type
      if not tt.HasField("shape"):
        # Ensure we set an empty list if the const is scalar (zero dims)
        tt.shape.dim.extend([])
        for dim in shape:
          tt.shape.dim.add().dim_value = dim
    # Handle subgraphs
    for node in graph.node:
      for attr in node.attribute:
        # Ref attrs refer to other attrs, so we don't need to do anything
        if attr.ref_attr_name != "":
          continue
        if 'AttributeProto' in str(type(attr.type)):
          add_const_value_infos_to_graph(attr.g)
          # if attr.type == AttributeProto.GRAPHS:
          for g in attr.graphs:
            add_const_value_infos_to_graph(g)

  return add_const_value_infos_to_graph(model.graph)


def make_deepcopy(graph, model):
  return deepcopy(graph), deepcopy(model)


def fallback_to_checkpoint(class_name, model, model_chkpt, G_chkpt):
  logger.exception("Failed in %s ..." % class_name)
  logger.warn("... Falling back to checkpoint graph/model")
  model.graph.CopyFrom(model_chkpt.graph)
  return deepcopy(G_chkpt)


def set_dict(dictionary: dict, key: str, value: Any):
  dictionary[key] = value


def set_dict_iterative(dictionary: dict, keys: list, values: list):
  if len(keys) != len(values):
    raise ValueError("Lengths of iterables are different")
  for key, value in zip(keys, values):
    dictionary[key] = value


def add_attribute(onnx_node, attribute_name, value):
  onnx_node.attribute.append(helper.make_attribute(attribute_name, value))


def rename_layers(node_names_list: List[str], name_mapping: dict, idx: int):
  for node_name in node_names_list:
    name_mapping[node_name] = node_name + f'_{idx}'
  return [name_mapping[old_name] for old_name in node_names_list]


def relabel_netx_graph(replacement_graph: nx.DiGraph, name_mapping: dict) -> nx.DiGraph:
  """
  Function to relabel networkx graph along with the node name present in the onnx node data 
  """
  for old_name, new_name in name_mapping.items():
    replacement_graph.nodes[old_name]['data'].name = new_name
  replacement_graph = nx.relabel_nodes(replacement_graph, name_mapping)
  return replacement_graph


class OnnxInference():

  def __init__(self):
    pass

  def inference_all_nodes(self, model_path, input_name, input_shape, onnx_backend="onnxruntime", images=None, data_format=None, dformat=None):
    '''
        Returns a list of tensors corresponding to per 
        layer output along with an index map
        {'output_node_name' : index}
        TODO : Multi input support        
        '''
    Args = namedtuple('Args', ['input_shape', 'data_format', 'input_node_name', 'dformat'])
    args_input_shape = [','.join([str(j) for j in i]) for i in input_shape]
    assert data_format is not None
    assert dformat is not None
    args_data_format = [i for i in data_format.split(' ')]
    args_dformat = [i for i in dformat.split(' ')]
    args = Args(args_input_shape, args_data_format, input_name, args_dformat)
    feed_dict = {}
    if images:
      try:
        feed_dict = io_utils.generate_input_raw(args, images)
      except Exception as e:
        feed_dict = io_utils.generate_input_default(args)
    else:
      feed_dict = io_utils.generate_input_default(args)
    feed_dict = list(feed_dict.values())[0]
    shape_dict = dict()
    if onnx_backend == "onnxruntime":
      session = onnxruntime.InferenceSession(model_path, sess_options=_sess_options)
      output_name = {}  #[n.name for n in session.get_outputs()]
      # for idx, n in enumerate(session.get_outputs()):
      #     output_name[n.name] = idx
      # out_layers = list(output_name.keys())
      out_layers = [n.name for n in session.get_outputs()]
      inf = session.run(None, feed_dict)
      for idx, v in enumerate(out_layers):
        if isinstance(inf[idx], list):
          shape_dict[v] = [list(arr.shape) if arr.size != () else [1] for arr in inf[idx]]
        else:
          shape_dict[v] = list(inf[idx].shape) if inf[idx].size != () else [1]

      for idx, inp in enumerate(session.get_inputs()):
        shape_dict[inp.name] = input_shape[idx]
    '''        
        # DEPRECATED: context: version upgrade for torch, onnx, onnxruntime
        elif onnx_backend == "caffe2": 
            model = onnx.load(model_path)
            prepared_backend = Caffe2Backend.prepare(model, no_check_UNSAFE=True)
            external_outputs = list()
            for node in prepared_backend.predict_net.op: 
                external_outputs.extend(node.output)
            prepared_backend.predict_net.external_output[:] = external_outputs 
            inf = prepared_backend.run(feed_dict)
            inf = inf._asdict()
            for k, v in inf.items():
                shape_dict[k] = v.shape
        '''
    return shape_dict

  def inference_output_nodes(self, model_path, input_shape, data=None, onnx_backend="onnxruntime"):
    ''' 
        Returns a list of tensors for the output nodes 
        in the graph.
        TODO : Multi input support 
        '''
    inf = None
    if data is None:
      data = np.random.random_sample(tuple(input_shape)).astype(np.float32)
    if onnx_backend == "onnxruntime":
      session = onnxruntime.InferenceSession(model_path, sess_options=_sess_options)
      input_name = session.get_inputs()[0].name
      output_name = [n.name for n in session.get_outputs()]
      inf = session.run(None, {input_name: data})
    elif onnx_backend == "caffe2":
      # TODO: Aditya
      pass
    return inf

  def get_output_shapes(self, model_path, all_inference_model, input_name, input_shape, onnx_backend="onnxruntime", images=None, data_format=None, dformat=None):
    shape_dict = self.inference_all_nodes(all_inference_model, input_name, input_shape, onnx_backend, images, data_format, dformat)
    return shape_dict

  def priorbox_inference(self, model_path, input_name, input_shape):
    ''' 
        Returns a list of tensors for the output node
        in the graph. Input data has all zeros. 
        '''
    inf = None
    if onnx_backend == "onnxruntime":
      data = np.zeros(tuple(input_shape)).astype(np.float32)
      session = onnxruntime.InferenceSession(model_path, sess_options=_sess_options)
      input_name = session.get_inputs()[0].name
      output_name = [n.name for n in session.get_outputs()]
      inf = session.run(None, {input_name: data})
    '''
        # DEPRECATED: context upgrade version of torch, onnx, onnxruntime
        elif onnx_backend == "caffe2": 
            model = onnx.load(model_path)
            # model = add_const_value_infos_to_graph(model)
            prepared_backend = Caffe2Backend.prepare(model)
            for idx, inp in enumerate(input_name): 
                feed_dict[inp] = np.zeros(tuple(input_shape)).astype(np.float32)
            inf = prepared_backend.run(feed_dict)
            inf = dict(inf._asdict())
        '''
    return inf


def real_variable_name(real_name):
  return real_name.replace('/', '_').replace('-', '_').replace('[', '_').replace(']', '_')


def topological_sort(G):
  return list(nx.lexicographical_topological_sort(G))


# returns a DiGraph given a path to a dot file
def read_dot(path):
  G = None
  try:
    G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(path))
  except Exception as e:
    logger.error("Error : %s", e)
  return G


def _output_tensor_to_name_map(onnx_model):
  _output_tensor_map = dict()
  for node in onnx_model.graph.input:
    _output_tensor_map[node.name] = node.name
  for node in onnx_model.graph.output:
    _output_tensor_map[node.name] = node.name
  for nodes in onnx_model.graph.node:
    for _idx, tensors in enumerate(nodes.output):
      _output_tensor_map[tensors] = nodes.name
  return _output_tensor_map


def _valueinfo_map(onnx_model):
  # assert len(list(onnx_model.graph.value_info)) > 0
  tensor_list = {}
  for _idx, nodes in enumerate(onnx_model.graph.value_info):
    tensor_list[nodes.name] = nodes
  for node in onnx_model.graph.output:
    tensor_list[node.name] = node
  return tensor_list


def add_node(G, node):
  G.add_node(node.name, data=node, type=node.op_type)


def add_initializer_node(G, node):
  G.add_node(node.name, data=node, type="tensor")


def add_input_node(G, onnx_model):
  for idx, node in enumerate(onnx_model.graph.input):
    G.add_node(node.name, data=node, type="input")


def add_output_node(G, onnx_model):
  _out_nodes = list()
  for idx, node in enumerate(onnx_model.graph.output):
    _out_nodes.append(node.name)
    G.add_node(node.name, data=node, type='output')
  return _out_nodes


def onnx_make_Graph(onnx_model):
  G = nx.DiGraph()
  edge_list = list()
  tensor_to_name_map = _output_tensor_to_name_map(onnx_model)
  add_input_node(G, onnx_model)
  for _idx, node in enumerate(onnx_model.graph.initializer):
    add_initializer_node(G, node)
  for _idx, node in enumerate(onnx_model.graph.node):
    add_node(G, node)
    for _inp in node.input:
      if _inp in tensor_to_name_map:
        _inp = tensor_to_name_map[_inp]
      logger.debug("Edge from - [%s] to [%s]", _inp, node.name)
      edge_list.append(tuple((_inp, node.name)))
  G.add_edges_from(edge_list)
  return G


def display_graph(G):
  import matplotlib.pyplot as plt
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, node_size=500)
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, arrows=True)
  plt.show()


def nx_graph_object(model, model_type="onnx"):
  return onnx_make_Graph(model)


def reverse_dict(dct):
  return dict((v, k) for k, v in dct.items())


def replace_node_with(nodes, node_name, new_node):
  """clears the data node with .name == node_name in nodes
    and copies all attributes to it from new_node
    """
  for idx, node in enumerate(nodes):
    if node.name == node_name:
      nodes[idx].Clear()
      nodes[idx].MergeFrom(new_node)


def get_predecessor(G, node):
  return list(G.predecessors(node))


def get_successors(G, node):
  return list(G.successors(node))


def remove_subgraph(G, remove_list):
  G.remove_nodes_from(remove_list)
  return G


def bfs_traversal(G, node, depth=None):
  bfs = dict(nx.bfs_successors(G, node, depth_limit=depth))
  desc = set(bfs.keys())
  desc.update(set(list(itertools.chain.from_iterable(list(bfs.values())))))
  return desc


def onnx_make_dummy_node(name, op_type="Constant", _type=None, _out=None, **kwargs):
  if _out is None:
    _out = [name]
  if _type is None:
    _type = []
  else:
    _type = [_type.name]
  node_def = helper.make_node(
      op_type,  # op_type
      _type,  # inputs
      _out,  # outputs,
      name=name,
      **kwargs)
  return node_def


def _proto_copy_attribute(copy_from, copy_to):
  copy_to.attribute.MergeFrom(copy_from.attribute)


# TODO unused code
# class NXtoGraph():

#   def __init__(self):
#     self.value_info = None
#     pass

#   ''' multi-inp not supported
#     '''

#   def nx_to_onnx(self, G, model, shape):
#     self.value_info = _valueinfo_map(model)
#     SG = ExtractSubgraph().extract_subgraph_from_source(G, list(G.nodes), [])
#     node_proto, initializers = ExtractSubgraph().extract_data_from_subgraph(SG, self.value_info)
#     inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, tuple(shape[0]))
#     out = helper.make_tensor_value_info('output', TensorProto.FLOAT, ())
#     graph = helper.make_graph(node_proto, 'nxgraph', inp, [out], initializer=initializers)
#     model_def = helper.make_model(graph, producer_name='onnx_example')
#     model_def.opset_import[0].version = model.opset_import[0].version
#     onnx.save(model_def, './nx_to_onnx.onnx')

#   def nx_to_onnx_graph_object(self, G, model, shape_dict, out_shapes, input_initializers, input_node, output_node, output_dir='.'):
#     self.value_info = _valueinfo_map(model)
#     SG = ExtractSubgraph().extract_subgraph_from_source(G, list(G.nodes), [])
#     node_p, inits = ExtractSubgraph().extract_data_from_subgraph(SG, self.value_info)
#     node_proto = copy.deepcopy(node_p)
#     initializers = copy.deepcopy(inits)
#     out = []
#     for i in output_node:
#       o = helper.make_tensor_value_info('output_' + i, TensorProto.FLOAT, tuple(out_shapes[i]))
#       out.append(o)
#     inp = []
#     for i in input_node:
#       inpu = helper.make_tensor_value_info('input_' + i, TensorProto.FLOAT, tuple(shape_dict['input_node_' + i]))
#       inp.append(inpu)
#     for i in input_initializers.keys():
#       for j in input_initializers[i].keys():
#         initializers.append(input_initializers[i][j])
#     for i in node_proto:
#       if i.name in input_node:
#         i.input[0] = 'input_' + i.name
#       if i.name in output_node:
#         i.output[0] = 'output_' + i.name
#     graph = helper.make_graph(node_proto, 'nxgraph', inp, out, initializer=initializers)
#     model_def = helper.make_model(graph, producer_name='onnx_example')
#     model_def.opset_import[0].version = model.opset_import[0].version
#     onnx.checker.check_model(model_def)
#     onnx.save(model_def, output_dir + '/nx_to_onnx.onnx')


class ExtractSubgraph():

  def __init__(self):
    pass

  def save_subgraph(self, G, path='./subgraph'):
    write_dot(G, "{}.dot".format(path))

  def relabel_nodes(self, G):
    _t_map = dict()
    relabel = dict()
    for idx, node in enumerate(G):
      G.nodes[node].pop('data')
    for idx, node in enumerate(G):
      if G.nodes[node]["type"] not in _t_map:
        _t_map[G.nodes[node]["type"]] = 0
      _t_map[G.nodes[node]["type"]] += 1
      new_node_name = "{}_{}".format(G.nodes[node]["type"], _t_map[G.nodes[node]["type"]])
      relabel[node] = new_node_name
    G = nx.relabel_nodes(G, relabel)

  # TODO Unused code
  # def extract_subgraph(self, model, input_node, output_node, path='./subgraph'):
  #   model_path = model
  #   onnx_model = onnx.load_model(model)
  #   onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
  #   G = onnx_to_nx(onnx_model)
  #   paths_between_node = nx.all_simple_paths(G, source=input_node, target=output_node)
  #   _nodes = {node for path in paths_between_node for node in path}
  #   SG = G.subgraph(nodes_between_set)
  #   SG = self.relabel_nodes(input_node, output_node)
  #   self.save_subgraph(input_node, output_node, path)

  def extract_subgraph_bw_nodes(self, G, model, input_nodes, output_nodes):  #, shape, path='./Subgraph'):
    nodes_bw_set = set()
    _subset = list(itertools.product(input_nodes, output_nodes))
    SG = G.copy()
    '''TODO : Correct logic for subgraph extraction. Huge time complexity in nx.all_simple_paths.
           Need to find efficient way to do this.
        for inp, out in _subset:
            paths = nx.all_simple_paths(G,inp, out)
            for path in paths:
                for nodes in path:
                    nodes_bw_set.add(nodes)
        for entry in model.graph.initializer:
            nodes_bw_set.add(entry.name)
        for nodes in G.nodes:
            if nodes not in nodes_bw_set:
                SG.remove_node(nodes)
        SG.remove_nodes_from(list(nx.isolates(SG)))
        '''
    for inp, out in _subset:
      ancestors_inp = nx.ancestors(SG, inp)
      desc_out = nx.descendants(SG, out)
      SG.remove_nodes_from(ancestors_inp)
      SG.remove_nodes_from(desc_out)
      SG.remove_nodes_from(list(nx.isolates(SG)))
    return SG

  def extract_subgraph_from_source(self, G, matched_nodes, exclude_set=[]):
    _all_nodes = set()
    for node in matched_nodes:
      x = get_predecessor(G, node)
      _all_nodes.add(node)
      for _x in x:
        if _x in exclude_set:
          continue
        _all_nodes.add(_x)
    return G.subgraph(_all_nodes).copy()

  def extract_data_from_subgraph(self, G, value_info, pop_output=None):
    top_sort = topological_sort(G)
    node_def = list()
    init = list()
    node_p = list()
    for node in top_sort:
      data = None
      if 'data' in G.nodes[node]:
        data = G.nodes[node]['data']
        node_def.append(data)
    for n in node_def:
      if 'TensorProto' in str(type(n)):
        init.append(n)
      elif 'NodeProto' in str(type(n)):
        if pop_output in n.output:
          n.output.remove(pop_output)
        node_p.append(n)
    # return  NodeProto, Initializer
    return node_p, init


class BaseMatcher():

  def __init__(self, onnx_model):
    self.tensor_to_name_map = _output_tensor_to_name_map(onnx_model)
    self.replacement_is_graph = False  #Set to True in the child class if replacement is a subgraph
    self.counter = 0

  def get_float_attribute(self, data_node: onnx.onnx_ml_pb2.NodeProto, name):
    ret = []
    for attr in data_node.attribute:
      if attr.name == name:
        return attr.f

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    pass

  def subgraph_to_replace(self) -> nx.DiGraph:
    """Returns replacement subgraph template"""
    pass

  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    pass

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    pass

  def get_replacement_output_node(self) -> str:
    """Returns the name of the output node in the replacement graph"""
    pass

  def get_input_to_replacement_graph_dict(self):
    """"Returns dictionary mapping node in replacement graph to corresponding node outside the replacement graph feeding input"""
    pass

  def update_counter(self):
    """Returns current value of counter and updates to ensure unique naming if there are multiple matches"""
    current_value = self.counter
    self.counter += 1
    return current_value

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove):
    """
    Constructs a replacement node
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: Dictionary matching template subgraph nodes to nodes in source graph
      model: ONNX Model
      inputs_to_collapsed: Predecessors to input node of collapsed graph
      nodes_to_remove: list of names of nodes that matched to the template subgraph and have to be removed
    Returns:
      replacement ONNX node
    """
    return None

  def mk_replacement_graph(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove):
    """
    Constructs a replacement graph
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: Dictionary matching template subgraph nodes to nodes in source graph
      model: ONNX Model
      inputs_to_collaphse: Predecessors to input node of collapsed graph
      nodes_to_remove: list of names of nodes that matched to the template subgraph and have to be removed
    Returns:
      replacement_graph: The netx subgraph to act as the replacement
    """
    return None

  def export_attributes(self, source_graph, nodes):
    _attributes = dict()
    for node in nodes:
      _attributes[node] = source_graph.nodes[node]["data"].attribute
    return _attributes

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    # Const* vs Constant
    if not n1 or not n2:
      return False
    x = (str.lower(n1t) == str.lower(n2t))
    return x

  def replace_subgraph_instances(self, source_graph, model):
    G_chkpt, model_chkpt = make_deepcopy(source_graph, model)
    try:
      self.value_info = _valueinfo_map(model)
      matches = isomorphism.DiGraphMatcher(source_graph, self.subgraph_to_match(), node_match=self.node_comparator_by_type)
      def _get_input_for_nodes(node_names):
        """for the given nodes, get the unique
                set of inputs to these nodes
                """
        names = list()
        for name in node_names:
          names.extend(source_graph.nodes[name]['data'].input)
        return list(dict.fromkeys(names))

      matched_nodes = set()
      for match in list(matches.subgraph_isomorphisms_iter()):
        match = reverse_dict(match)
        iso_nodes = set(list(match.values()))
        if matched_nodes.intersection(iso_nodes) == iso_nodes:
          continue
        matched_nodes.update(iso_nodes)
        nodes_to_remove = list(match.values())
        inputs_to_collapsed_node =\
                list(_get_input_for_nodes(map(match.get, self.input_nodes())))
        _inputs = list()
        for inp in inputs_to_collapsed_node:
          _inputs.append(inp)
        # attributes of nodes which will be replaced
        nodes = matched_nodes
        # attr_map = self.export_attributes(source_graph, nodes)
        name_of_node = match[self.output_node()]
        # copy over the shape data
        # TODO: refactor into a function that copies essential
        # attributes
        inp_nodes_netx = [i for i in map(match.get, self.input_nodes())]
        inp_collaped_nodes_netx = set()
        for i in inp_nodes_netx:
          for j in source_graph.predecessors(i):
            inp_collaped_nodes_netx.add(j)
        inp_collaped_nodes_netx = [i for i in inp_collaped_nodes_netx if i not in nodes_to_remove]

        if hasattr(self, 'replacement_is_graph') and self.replacement_is_graph:  #We are going to replace matched subgraph with another subgraph
          replacement_graph = self.mk_replacement_graph(source_graph, match, model, inputs_to_collapsed_node, nodes_to_remove)
          # replacement_graph == None ==> some extra condition in mk_replacement_graph is not satisfied.
          # * don't replace. !!
          if replacement_graph == None:
            continue
          output_landing_nodes = list(source_graph.successors(match[self.output_node()]))
          output_tensor = source_graph.nodes[match[self.output_node()]]['data'].output[0]  #Storing output tensor name of previous output node before nodes are removed
          source_graph = remove_subgraph(source_graph, nodes_to_remove)
          logger.debug("Nodes removed: %s", " ".join(nodes_to_remove))
          #Replacing inputs of nodes in source graph whose successor is now inside the replacement graph
          new_output_edges = []
          for node in output_landing_nodes:
            new_input_list = []
            for predecessor_input in source_graph.nodes[node]['data'].input:
              if predecessor_input == output_tensor:
                new_input = replacement_graph.nodes[self.get_replacement_output_node()]['data'].output[0]
                new_input_list.append(new_input)
                new_output_edges.append((self.get_replacement_output_node(), node))
              else:
                new_input_list.append(predecessor_input)
            source_graph.nodes[node]['data'].input[:] = new_input_list
          for node in replacement_graph.nodes.values():
            #Ask whether we should compare it with nodes to remove
            # Should I raise error if data column is missing?
            source_graph.add_node(node['data'].name, data=node['data'], type=node['type'])
          source_graph.add_edges_from(new_output_edges)
          #Adding input to replacement graph
          for key, value in self.get_input_to_replacement_graph_dict().items():
            source_graph.add_edge(value, key)
          for node1, node2 in replacement_graph.edges():
            source_graph.add_edge(node1, node2)
          logger.info("Subgraph successfully replaced.")
          logger.info("Node info", source_graph.nodes)
        else:  #We are going to replace matched graph with a node
          replacement_node = self.mk_replacement_node(source_graph, match, model, inputs_to_collapsed_node, nodes_to_remove)
          # replacement_node == None ==> some extra condition in mk_replacement_node is not satisfied.
          # * don't replace. !!x
          if replacement_node == None:
            continue
          output_landing_nodes = list(source_graph.successors(match[self.output_node()]))
          replace_node_with(model.graph.node, replacement_node.name, replacement_node)
          logger.debug("Node info", source_graph.nodes)
          # only delete the old nodes from the graphs once the new entry is
          # made as the new graph node _might_ use information from the nodes
          # that are going to be deleted from the graph.
          logger.debug("Nodes removed: %s", " ".join(nodes_to_remove))
          source_graph = remove_subgraph(source_graph, nodes_to_remove)
          source_graph.add_node(replacement_node.name, data=replacement_node, type=replacement_node.op_type)
          new_edges=\
                  [(input_node, name_of_node) for input_node in inp_collaped_nodes_netx]
          source_graph.add_edges_from(new_edges)
          output_edges=\
                  [(name_of_node,output_landing_node) for output_landing_node in output_landing_nodes]
          source_graph.add_edges_from(output_edges)
          logger.info("Node successfully replaced. Replaced node: %s, Type: %s", replacement_node.name, replacement_node.op_type)
          logger.info("Node info", source_graph.nodes)
    except Exception as e:
      source_graph = fallback_to_checkpoint(self.__class__.__name__, model, model_chkpt, G_chkpt)
    # write_dot_colon_wrapper(source_graph, 'full_graph_rms.dot')
    return source_graph


class DetectionBoxMatcher():

  def __init__(self, path, shape_dict):
    self.path = path
    self.G = None
    self.value_info = dict()
    self.shape_dict = shape_dict  # OnnxInference().get_output_shapes(self.path, input_shape)
    self.tensor_to_name_map = None

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    # Const* vs Constant
    if not n1 or not n2:
      return False
    return n1t == n2t

  def subgraph_to_match(self):
    self.G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.path))
    return self.G

  def input_nodes(self):
    return [node for node in self.G.nodes if self.G.in_degree(node) == 0]

  def output_node(self):
    return [node for node in self.G.nodes if self.G.out_degree(node) == 0][0]

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "DetectionBoxOutput", _type=None, _out=_data.output)
    inputs_to_collapsed.sort()
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = []
    _proto_copy_attribute(_data, node_def)
    return node_def

  def get_subgraph_inferece(self, path, input_name, input_shape):
    _t = OnnxInference().priorbox_inference(path, input_name, input_shape)
    return _t[0]

  def replace_subgraph_instances(self, source_graph, model):
    G_chkpt, model_chkpt = make_deepcopy(source_graph, model)
    try:
      self.value_info = _valueinfo_map(model)
      self.tensor_to_name_map = _output_tensor_to_name_map(model)
      matches = isomorphism.DiGraphMatcher(source_graph, self.subgraph_to_match(), node_match=self.node_comparator_by_type)

      def _get_input_for_nodes(node_names):
        """for the given nodes, get the unique
                set of inputs to these nodes
                """
        names = set()
        for name in node_names:
          names.update(source_graph.nodes[name]['data'].input)
        return names

      matched_nodes = set()
      for match in list(matches.subgraph_isomorphisms_iter()):
        match = reverse_dict(match)
        iso_nodes = set(list(match.values()))
        if matched_nodes.intersection(iso_nodes) == iso_nodes:
          continue
        matched_nodes.update(iso_nodes)
        nodes_to_remove = list(match.values())
        inputs_to_collapsed_node =\
                list(_get_input_for_nodes(map(match.get, self.input_nodes())))
        _inputs = list()
        for inp in inputs_to_collapsed_node:
          if inp in self.tensor_to_name_map[inp]:
            _inputs.append(self.tensor_to_name_map[inp])
          else:
            _inputs.append(inp)
        # attributes of nodes which will be replaced
        # nodes = matched_nodes
        # attr_map = self.export_attributes(source_graph, nodes)
        matched_nodes = match.values()
        # model_output_node = match[self.output_node()]
        model_input_node = self.input_nodes()
        parent_of_input_node = list()
        for n in model_input_node:
          model_input_node = match[n]
          _x = get_predecessor(source_graph, model_input_node)
          parent_of_input_node.extend(_x)
        SG = ExtractSubgraph().extract_subgraph_from_source(source_graph, matched_nodes, parent_of_input_node)
        _nodes_to_remove = list(SG.node)
        name_of_node = match[self.output_node()]
        inputs_to_collapsed_node = [i for i in inputs_to_collapsed_node if i not in nodes_to_remove]
        replacement_node = self.mk_replacement_node(source_graph, match, model, _inputs, _nodes_to_remove)
        # replacement_node == None ==> some extra condition in mk_replacement_node is not satisfied.
        # * don't replace. !!
        if replacement_node == None:
          continue
        # copy over the shape data
        # TODO: refactor into a function that copies essential
        # attributes
        replace_node_with(model.graph.node, replacement_node.name, replacement_node)
        # only delete the old nodes from the graphs once the new entry is
        # made as the new graph node _might_ use information from the nodes
        # that are going to be deleted from the graph.
        source_graph = remove_subgraph(source_graph, _nodes_to_remove)
        source_graph.add_node(replacement_node.name, data=replacement_node, type=replacement_node.op_type)
        new_edges =\
                [(input_node, name_of_node) for input_node in inputs_to_collapsed_node]
        source_graph.add_edges_from(new_edges)
        logger.info("Node successfully replaced. Replaced node: %s, Type: %s", replacement_node.name, replacement_node.op_type)
    except Exception as e:
      source_graph = fallback_to_checkpoint(self.__class__.__name__, model, model_chkpt, G_chkpt)
    return source_graph


class PriorBoxDotMatcher():

  def __init__(self, path, shape_dict, state_dict):
    self.path = path
    self.G = None
    self.priorbox_inference = dict()
    self.value_info = dict()
    self.shape_dict = shape_dict  # OnnxInference().get_output_shapes(self.path, input_shape)
    self.tensor_to_name_map = None
    self.state_dict = state_dict

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    # Const* vs Constant
    if not n1 or not n2:
      return False
    # x = re.match(n2t+"*", n1t)
    # if x == None :
    #     return False
    # return True
    return n1t == n2t

  def subgraph_to_match(self):
    self.G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.path))
    return self.G

  def input_nodes(self):
    assert self.G is not None
    return [node for node in self.G.nodes if self.G.in_degree(node) == 0]

  def output_node(self):
    assert self.G is not None
    return [node for node in self.G.nodes if self.G.out_degree(node) == 0][0]

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Reshape", _type=self.value_info[_data.output[0]], _out=_data.output)
    # need a clean solution.
    inputs_to_collapsed.sort()
    node_def.input[:] = inputs_to_collapsed
    _proto_copy_attribute(_data, node_def)
    return node_def

  def get_subgraph_inferece(self, path, input_name, input_shape):
    _t = OnnxInference().priorbox_inference(path, input_name, input_shape)
    return _t[0]

  def replace_subgraph_instances(self, source_graph, model):
    matches = isomorphism.DiGraphMatcher(source_graph, self.subgraph_to_match(), node_match=self.node_comparator_by_type)
    self.value_info = _valueinfo_map(model)
    self.tensor_to_name_map = _output_tensor_to_name_map(model)

    def _get_input_for_nodes(node_names):
      """for the given nodes, get the unique
            set of inputs to these nodes
            """
      names = set()
      for name in node_names:
        names.update(source_graph.nodes[name]['data'].input)
      return names

    _matched_nodes = set()
    concat_node = None
    for match in list(matches.subgraph_isomorphisms_iter()):
      match = reverse_dict(match)
      iso_nodes = set(list(match.values()))
      if _matched_nodes.intersection(iso_nodes) == iso_nodes:
        continue
      _matched_nodes.update(iso_nodes)
      nodes_to_remove = list(match.values())
      inputs_to_collapsed_node =\
              list(_get_input_for_nodes(map(match.get, self.input_nodes())))
      _inputs = list()
      # _inputs = [x for x in inputs_to_collapsed if x in ]
      for inp in inputs_to_collapsed_node:
        if inp in self.tensor_to_name_map[inp]:
          _inputs.append(self.tensor_to_name_map[inp])
        else:
          _inputs.append(inp)
      # attributes of nodes which will be replaced
      # attr_map = self.export_attributes(source_graph, nodes)
      name_of_node = match[self.output_node()]
      inputs_to_collapsed_node = [i for i in inputs_to_collapsed_node if i not in nodes_to_remove]
      # get inference from the subgraph
      matched_nodes = match.values()
      model_output_node = match[self.output_node()]
      model_input_node = match[self.input_nodes()[0]]
      parent_of_input_node = get_predecessor(source_graph, model_input_node)
      source_node_parent_tensor = source_graph.nodes[parent_of_input_node[0]]['data'].output
      output_node_parent_tensor = source_graph.nodes[model_output_node]['data'].output
      # # self.tensor_to_name_map
      # rev_name_map = reverse_dict(self.tensor_to_name_map)
      # if model_input_node in rev_name_map:
      #     model_input_node = rev_name_map[model_input_node]
      shape_of_inp_node = self.shape_dict[source_node_parent_tensor[0]]
      SG = ExtractSubgraph().extract_subgraph_from_source(source_graph, matched_nodes, [parent_of_input_node[0]])
      _nodes_to_remove = list(SG.node)
      inp = self.value_info[source_node_parent_tensor[0]]
      out = self.value_info[output_node_parent_tensor[0]]
      # nodeproto, initializer = ExtractSubgraph().extract_data_from_subgraph(SG, self.value_info ,source_node_parent_tensor[0])
      # shape_of_inp_node = [1]*(4-len(shape_of_inp_node))+shape_of_inp_node
      # inp = helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, shape_of_inp_node)
      # graph = helper.make_graph(
      #         nodeproto,
      #         'priorbox',
      #         [inp],
      #         [out],
      #         initializer=initializer
      #         )
      # model_def = helper.make_model(graph, producer_name='onnx_example')
      # onnx.save(model_def, './_priorbox.onnx')
      # self.priorbox_inference[model_output_node] = self.get_subgraph_inferece('./_priorbox.onnx', shape_of_inp_node)
      slice_nodes = [x for x, y in SG.nodes(data=True) if y['type'] == 'Slice']
      pbox_order = {}
      for _node in slice_nodes:
        get_succ = bfs_traversal(SG, _node, depth=3)
        init_pbox = [n for n in get_succ if SG.node[n]['type'] == 'Expand']
        init_pbox = SG.node[init_pbox[0]]["data"].input[0]
        starts = self.state_dict[SG.node[_node]["data"].input[1]]
        pbox_order[starts[0]] = self.state_dict[init_pbox]
      _get_child = get_successors(source_graph, model_output_node)
      concat_node = source_graph.nodes[_get_child[0]]['data']
      order_seq = []
      for r in range(0, 4):
        order_seq.append(pbox_order[r])
      pbox_out = np.concatenate(order_seq, axis=1)
      self.priorbox_inference[model_output_node] = pbox_out
      # need to pass --
      #               source node name
      #               all nodes present in subgraph
      #               input shape
      #
      # copy over the shape data
      # TODO: refactor into a function that copies essential
      # attributes
      replacement_node = self.mk_replacement_node(source_graph, match, model, _inputs, _nodes_to_remove)
      # replacement_node == None ==> some extra condition in mk_replacement_node is not satisfied.
      # * don't replace. !!
      if replacement_node == None:
        continue
      replace_node_with(model.graph.node, replacement_node.name, replacement_node)
      # only delete the old nodes from the graphs once the new entry is
      # made as the new graph node _might_ use information from the nodes
      # that are going to be deleted from the graph.
      source_graph = remove_subgraph(source_graph, _nodes_to_remove)
      source_graph.add_node(replacement_node.name, data=replacement_node, type=replacement_node.op_type)
      new_edges =\
              [(input_node, name_of_node) for input_node in inputs_to_collapsed_node]
      source_graph.add_edges_from(new_edges)
      logger.info("Node successfully replaced. Replaced node: %s, Type: %s", replacement_node.name, replacement_node.op_type)
    if concat_node is None:
      return source_graph, None
    _outputs = list()
    _output_order = list()
    _output_order = concat_node.input
    _output_order = [self.tensor_to_name_map[n] for n in _output_order]
    return source_graph, _output_order

  def get_priorbox_outputs(self, source_graph, model):
    G_chkpt, model_chkpt = make_deepcopy(source_graph, model)
    try:
      source_graph, output_order = self.replace_subgraph_instances(source_graph, model)
      if output_order is not None:
        self.priorbox_inference['output_order'] = output_order
        with open('priorbox_output.pickle', 'wb') as handle:
          pickle.dump(self.priorbox_inference, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
      source_graph = fallback_to_checkpoint(self.__class__.__name__, model, model_chkpt, G_chkpt)
    return source_graph


class BBoxTransformDotMatcher(BaseMatcher):

  def __init__(self, path, shape_dict, state_dict):
    self.path = path
    self.G = None
    # self.priorbox_inference = dict()
    self.value_info = dict()
    self.shape_dict = shape_dict  # OnnxInference().get_output_shapes(self.path, input_shape)
    self.tensor_to_name_map = None
    self.state_dict = state_dict

  def get_float_attribute(self, data_node: onnx.onnx_ml_pb2.NodeProto, name):
    ret = []
    for attr in data_node.attribute:
      if attr.name == name:
        return attr.f

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    # Const* vs Constant
    if not n1 or not n2:
      return False
    return n1t == n2t

  def subgraph_to_match(self):
    self.G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.path))
    if self.G.has_node("\\n"):
      self.G.remove_node("\\n")
    return self.G

  def input_nodes(self):
    return ["Reshape_16", "Reshape_56"]

  def output_node(self):
    return [node for node in self.G.nodes if self.G.out_degree(node) == 0][0]

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "BBoxTransform_detectron2", _type=self.value_info[_data.output[0]], _out=_data.output)
    clip_nodes = []
    for x, y in source_graph.nodes(data=True):
      if 'type' in y and y['type'] == 'Clip':
        clip_nodes.append(y['data'])
    scale_clamp = 1.7958800173  #defult value from caffe2 implementation
    for node in clip_nodes:
      if node.input[1] == "":
        scale_clamp = self.state_dict[node.input[2]].item()
    value = helper.make_attribute('scale_clamp', scale_clamp)
    node_def.attribute.MergeFrom([value])
    weights = []
    div_nodes = []
    for x, y in source_graph.nodes(data=True):
      if 'type' in y and y['type'] == 'Div':
        div_nodes.append(x)
    div_nodes.sort()
    weights = [self.state_dict[source_graph.nodes[node]['data'].input[1]].item() for node in div_nodes]
    value = helper.make_attribute('weights', weights)
    node_def.attribute.MergeFrom([value])
    # need a clean solution.
    inputs = []
    for node in inputs_to_collapsed:
      if node not in self.state_dict:
        inputs.append(node)
    node_def.input[:] = inputs
    _proto_copy_attribute(_data, node_def)
    # clip parms
    clip_param = [0.0] * 4
    clip_bbox = False
    min_nodes = []
    for k, v in matched_subgraph.items():
      y = source_graph.nodes[v]
      if 'type' in y and y['type'] == 'Min':
        min_nodes.append(v)
    initializers = set()
    for node in min_nodes:
      if source_graph.nodes[node]['data'].input[1] in self.state_dict:
        clip_bbox = True
        clip_param = self.state_dict[source_graph.nodes[node]['data'].input[1]].flatten()
        clip_param = clip_param.tolist()
    value = helper.make_attribute('clip_param', clip_param)
    node_def.attribute.MergeFrom([value])
    value = helper.make_attribute('clip_bbox', clip_bbox)
    node_def.attribute.MergeFrom([value])
    return node_def


class BBoxTransformDotStaticMatcher(BBoxTransformDotMatcher):

  def __init__(self, path, shape_dict, state_dict):
    self.path = path
    self.G = None
    # self.priorbox_inference = dict()
    self.value_info = dict()
    self.shape_dict = shape_dict  # OnnxInference().get_output_shapes(self.path, input_shape)
    self.state_dict = state_dict

  def input_nodes(self):
    assert self.G is not None
    return [node for node in self.G.nodes if self.G.in_degree(node) == 0]

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "BBoxTransform_static_detectron2", _type=self.value_info[_data.output[0]], _out=_data.output)
    clip_nodes = []
    for x, y in source_graph.nodes(data=True):
      if 'type' in y and y['type'] == 'Clip':
        clip_nodes.append(y['data'])
    scale_clamp = 1.7958800173  #defult value from caffe2 implementation
    for node in clip_nodes:
      if node.input[1] == "":
        scale_clamp = self.state_dict[node.input[2]].item()
    value = helper.make_attribute('scale_clamp', scale_clamp)
    node_def.attribute.MergeFrom([value])
    weights = []
    div_nodes = []
    for k, v in matched_subgraph.items():
      y = source_graph.nodes[v]
      if 'type' in y and y['type'] == 'Div':
        div_nodes.append(v)
    div_nodes.sort(reverse=True)
    weights = [self.state_dict[source_graph.nodes[node]['data'].input[1]].item() for node in div_nodes]
    value = helper.make_attribute('weights', weights)
    node_def.attribute.MergeFrom([value])
    inputs = [node for node in inputs_to_collapsed if node not in self.state_dict]
    # slice start determines where width, height , ctr_x and ctr_y comes from
    # start = 0 is width
    # start = 1 is height
    # start = 2 is ctr_x
    # start = 3 is ctr_y
    slice_nodes = []
    width = None
    height = None
    ctr_x = None
    ctr_y = None
    for k, v in matched_subgraph.items():
      y = source_graph.nodes[v]
      if 'type' in y and y['type'] == 'Slice':
        start = [int(d) for d in source_graph.nodes[y['data'].input[1]]['data'].raw_data][0]
        if start == 0:
          width = source_graph.nodes[get_successors(source_graph, get_successors(source_graph, v)[0])[0]]['data'].input[1]
          ctr_x = source_graph.nodes[get_successors(source_graph, get_successors(source_graph, get_successors(source_graph, v)[0])[0])[0]]['data'].input[1]
        elif start == 1:
          height = source_graph.nodes[get_successors(source_graph, get_successors(source_graph, v)[0])[0]]['data'].input[1]
          ctr_y = source_graph.nodes[get_successors(source_graph, get_successors(source_graph, get_successors(source_graph, v)[0])[0])[0]]['data'].input[1]
    for k, v in matched_subgraph.items():
      y = source_graph.nodes[v]
    inputs.extend([width, height, ctr_x, ctr_y])
    node_def.input[:] = inputs
    _proto_copy_attribute(_data, node_def)
    return node_def

  def replace_subgraph_instances(self, source_graph, model):
    G_chkpt, model_chkpt = make_deepcopy(source_graph, model)
    try:
      matches = isomorphism.DiGraphMatcher(source_graph, self.subgraph_to_match(), node_match=self.node_comparator_by_type)
      self.value_info = _valueinfo_map(model)
      self.tensor_to_name_map = _output_tensor_to_name_map(model)

      def _get_input_for_nodes(node_names):
        """for the given nodes, get the unique
                set of inputs to these nodes
                """
        names = set()
        for name in node_names:
          names.update(source_graph.nodes[name]['data'].input)
        return names

      _matched_nodes = set()
      concat_node = None
      for match in list(matches.subgraph_isomorphisms_iter()):
        match = reverse_dict(match)
        iso_nodes = set(list(match.values()))
        if _matched_nodes.intersection(iso_nodes) == iso_nodes:
          continue
        _matched_nodes.update(iso_nodes)
        nodes_to_remove = list(match.values())
        inputs_to_collapsed_node =\
                list(_get_input_for_nodes(map(match.get, self.input_nodes())))
        _inputs = inputs_to_collapsed_node
        name_of_node = match[self.output_node()]
        matched_nodes = match.values()
        model_output_node = match[self.output_node()]
        model_input_node = match[self.input_nodes()[0]]
        parent_of_input_node = get_predecessor(source_graph, model_input_node)
        # copy over the shape data
        # TODO: refactor into a function that copies essential
        # attributes
        replacement_node = self.mk_replacement_node(source_graph, match, model, _inputs, nodes_to_remove)
        # replacement_node == None ==> some extra condition in mk_replacement_node is not satisfied.
        # * don't replace. !!
        if replacement_node == None:
          continue
        replace_node_with(model.graph.node, replacement_node.name, replacement_node)
        logger.debug("Node info", source_graph.nodes)
        # only delete the old nodes from the graphs once the new entry is
        # made as the new graph node _might_ use information from the nodes
        # that are going to be deleted from the graph.
        logger.debug("Nodes removed: %s", " ".join(nodes_to_remove))
        source_graph = remove_subgraph(source_graph, nodes_to_remove)
        source_graph.add_node(replacement_node.name, data=replacement_node, type=replacement_node.op_type)
        new_edges=\
                [(input_node, name_of_node) for input_node in inputs_to_collapsed_node]
        source_graph.add_edges_from(new_edges)
        logger.info("Node successfully replaced. Replaced node: %s, Type: %s", replacement_node.name, replacement_node.op_type)
        logger.info("Node info", source_graph.nodes)
    except Exception as e:
      source_graph = fallback_to_checkpoint(self.__class__.__name__, model, model_chkpt, G_chkpt)
    return source_graph


class TestMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    conv = onnx_make_dummy_node('onnx/basic', 'Conv')
    for node in [conv]:
      G.add_node(node.name, data=node, type=node.op_type)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/basic']

  def output_node(self):
    return "onnx/basic"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], 'Conv')
    node_def.input[:] = inputs_to_collapsed
    node_def.output.extend(_data.output)
    _proto_copy_attribute(_data, node_def)
    return node_def


class L2NormMatcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    cast = onnx_make_dummy_node('onnx/cast', 'Cast')
    shape = onnx_make_dummy_node('onnx/shape', 'Shape')
    _pow = onnx_make_dummy_node('onnx/pow', 'Pow')
    exp = onnx_make_dummy_node('onnx/expand', 'Expand')
    rsum = onnx_make_dummy_node('onnx/rsum', 'ReduceSum')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    add = onnx_make_dummy_node('onnx/add', 'Add')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [cast, shape, _pow, exp, rsum, mul, sqrt, add, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/cast', 'onnx/pow')))
    edge_list.append(tuple(('onnx/cast', 'onnx/shape')))
    edge_list.append(tuple(('onnx/cast', 'onnx/mul')))
    edge_list.append(tuple(('onnx/shape', 'onnx/expand')))
    edge_list.append(tuple(('onnx/expand', 'onnx/mul')))
    edge_list.append(tuple(('onnx/pow', 'onnx/rsum')))
    edge_list.append(tuple(('onnx/rsum', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/add')))
    edge_list.append(tuple(('onnx/add', 'onnx/div')))
    edge_list.append(tuple(('onnx/mul', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/cast']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    _const = source_graph.nodes[matched_subgraph['onnx/expand']]['data']
    init = _const.input[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm_SSD", _type=self.value_info[_data.output[0]], _out=_data.output)
    inputs_to_collapsed.append(init)
    node_def.input[:] = inputs_to_collapsed
    _proto_copy_attribute(_data, node_def)
    return node_def


class L2NormMatcher_zt(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    #relu = onnx_make_dummy_node('onnx/relu', 'Relu')
    #shape = onnx_make_dummy_node('onnx/shape', 'Shape')
    _pow = onnx_make_dummy_node('onnx/pow', 'Pow')
    #exp = onnx_make_dummy_node('onnx/expand', 'Expand')
    rsum = onnx_make_dummy_node('onnx/rsum', 'ReduceSum')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    add = onnx_make_dummy_node('onnx/add', 'Add')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [_pow, rsum, mul, sqrt, add, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/pow', 'onnx/rsum')))
    edge_list.append(tuple(('onnx/rsum', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/add')))
    edge_list.append(tuple(('onnx/add', 'onnx/div')))
    #edge_list.append(tuple(('onnx/relu', 'onnx/div')))
    #edge_list.append(tuple(('onnx/div', 'onnx/shape')))
    edge_list.append(tuple(('onnx/div', 'onnx/mul')))
    #edge_list.append(tuple(('onnx/shape', 'onnx/expand')))
    #edge_list.append(tuple(('onnx/expand', 'onnx/mul')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/pow']

  def output_node(self):
    return "onnx/mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    #print("IN pattren matcher \n",source_graph.nodes[matched_subgraph['onnx/mul']]['data'].input)
    #_const = source_graph.nodes[matched_subgraph['onnx/expand']]['data']
    _const = source_graph.nodes[matched_subgraph['onnx/mul']]['data']
    init = _const.input[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm_SSD", _type=self.value_info[_data.output[0]], _out=_data.output)
    inputs_to_collapsed.append(init)
    node_def.input[:] = inputs_to_collapsed
    _proto_copy_attribute(_data, node_def)
    return node_def


class L1NormMatcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    l2_reduce = onnx_make_dummy_node('onnx/reduceL1', 'ReduceL1')
    exp = onnx_make_dummy_node('onnx/expand', 'Expand')
    clip = onnx_make_dummy_node('onnx/clip', 'Clip')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [l2_reduce, exp, clip, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reduceL1', 'onnx/clip')))
    edge_list.append(tuple(('onnx/clip', 'onnx/expand')))
    edge_list.append(tuple(('onnx/expand', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reduceL1']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    _const = source_graph.nodes[matched_subgraph['onnx/div']]['data']
    init = _const.input[0]
    eps = self.state_dict[source_graph.nodes[matched_subgraph['onnx/clip']]['data'].input[1]].item(0)
    axes_attr = next(i for i in source_graph.nodes[matched_subgraph['onnx/reduceL1']]['data'].attribute if 'axes' == i.name)
    axes = axes_attr.ints
    input_shape = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/reduceL1']]['data'].input[0]]
    num_dims = len(input_shape)
    axes = [i + num_dims if i < 0 else i for i in axes]

    axis = 1
    if (len(axes) == 1 and axes[0] == 1):
      axis = axes[0]
    elif (set(range(num_dims)) - set(axes) == set([0, 1])):
      axis = 5
    else:
      raise Exception(f"L1NormMatcher does not support axes: {axes}")

    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L1Norm", _eps=eps, _axes=axis, _type=self.value_info[_data.output[0]], _out=_data.output)
    inputs_to_collapsed.append(init)
    node_def.input[:] = inputs_to_collapsed
    _proto_copy_attribute(_data, node_def)
    return node_def


class L2NormMatcher_AC(BaseMatcher):

  def __init__(self, model, shape_dict, state_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    _pow = onnx_make_dummy_node('onnx/pow', 'Pow')
    rsum = onnx_make_dummy_node('onnx/rsum', 'ReduceSum')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    add = onnx_make_dummy_node('onnx/add', 'Add')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [_pow, rsum, sqrt, add, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/pow', 'onnx/rsum')))
    edge_list.append(tuple(('onnx/rsum', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/add')))
    edge_list.append(tuple(('onnx/add', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/pow']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    _const = source_graph.nodes[matched_subgraph['onnx/div']]['data']
    init = _const.input[0]
    _rsum = source_graph.nodes[matched_subgraph['onnx/rsum']]['data'].output[0]
    rshape = self.shape_dict[_rsum]
    if np.prod(rshape) != 1:
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm")
      node_def.attribute.append(helper.make_attribute("across_spatial", False))
      node_def.input[:] = inputs_to_collapsed
      node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    else:
      eps = self.state_dict[source_graph.nodes[matched_subgraph['onnx/add']]['data'].input[1]].item(0)
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm_AC", _eps=eps, _type=self.value_info[_data.output[0]], _out=_data.output)
      inputs_to_collapsed.append(init)
      node_def.input[:] = inputs_to_collapsed
      _proto_copy_attribute(_data, node_def)
    return node_def


class L2NormScale_Matcher(BaseMatcher):

  def __init__(self, model, shape_dict, state_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    rsum = onnx_make_dummy_node('onnx/rsum', 'ReduceSum')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    mul2 = onnx_make_dummy_node('onnx/mul2', 'Mul')
    clip = onnx_make_dummy_node('onnx/clip', 'Clip')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    mul3 = onnx_make_dummy_node('onnx/mul3', 'Mul')
    for node in [mul, rsum, sqrt, mul2, clip, div, mul3]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/mul', 'onnx/rsum')))
    edge_list.append(tuple(('onnx/rsum', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/mul2')))
    edge_list.append(tuple(('onnx/mul2', 'onnx/clip')))
    edge_list.append(tuple(('onnx/clip', 'onnx/div')))
    edge_list.append(tuple(('onnx/div', 'onnx/mul3')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul']

  def output_node(self):
    return "onnx/mul3"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):

    def get_constant(val):
      if isinstance(val, float) or isinstance(val, int):
        return val
      elif isinstance(val, list) and len(val) == 1:
        return get_constant(val[0])
      else:
        return None

    inputs_to_mul = source_graph.nodes[matched_subgraph['onnx/mul']]['data'].input
    inputs_to_div = source_graph.nodes[matched_subgraph['onnx/div']]['data'].input

    # Check onnx/mul have same 2 inputs
    if len(inputs_to_mul) != 2 and len(set(inputs_to_mul)) != 1:
      return None

    # Check onnx/mul and onnx/div have same inputs
    if len(set(inputs_to_mul).intersection(set(inputs_to_div))) == 0:
      return None

    # Get scalar value from mul2 node
    temp = self.state_dict[source_graph.nodes[matched_subgraph['onnx/mul2']]['data'].input[1]].tolist()
    temp = get_constant(temp)
    if temp is None:
      return None
    scale_before_clip = temp

    # Get scalar value from mul3 node
    temp = self.state_dict[source_graph.nodes[matched_subgraph['onnx/mul3']]['data'].input[1]].tolist()
    temp = get_constant(temp)
    if temp is None:
      return None
    scale_after_div = temp

    # Get eps value from clip node
    temp = self.state_dict[source_graph.nodes[matched_subgraph['onnx/clip']]['data'].input[1]].tolist()
    temp = get_constant(temp)
    if temp is None:
      eps = 0.000009999  # Default value for eps
    eps = temp

    # Node creation
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm_scale")
    node_def.attribute.append(helper.make_attribute("across_spatial", True))
    node_def.attribute.append(helper.make_attribute("eps", eps))
    node_def.attribute.append(helper.make_attribute("scale_before_clip", scale_before_clip))
    node_def.attribute.append(helper.make_attribute("scale_after_div", scale_after_div))

    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    return node_def


class Centernet_IR1_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    cast1 = onnx_make_dummy_node('onnx/cast_1', 'Cast')
    cast2 = onnx_make_dummy_node('onnx/cast_2', 'Cast')
    gather = onnx_make_dummy_node('onnx/gather', 'Gather')
    slice1 = onnx_make_dummy_node('onnx/slice_1', 'Slice')
    slice2 = onnx_make_dummy_node('onnx/slice_2', 'Slice')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    shape = onnx_make_dummy_node('onnx/shape', 'Shape')
    concat = onnx_make_dummy_node('onnx/concat', 'Concat')
    resize = onnx_make_dummy_node('onnx/resize', 'Resize')
    for node in [cast1, cast2, gather, slice1, slice2, mul, shape, concat, resize]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/shape', 'onnx/gather')))
    edge_list.append(tuple(('onnx/gather', 'onnx/cast_1')))
    edge_list.append(tuple(('onnx/cast_1', 'onnx/slice_1')))
    edge_list.append(tuple(('onnx/slice_1', 'onnx/mul')))
    edge_list.append(tuple(('onnx/mul', 'onnx/cast_2')))
    edge_list.append(tuple(('onnx/cast_2', 'onnx/concat')))
    edge_list.append(tuple(('onnx/shape', 'onnx/slice_2')))
    edge_list.append(tuple(('onnx/slice_2', 'onnx/concat')))
    edge_list.append(tuple(('onnx/concat', 'onnx/resize')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/shape']

  def output_node(self):
    return "onnx/resize"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Centernet_IR1", _type=self.value_info[_data.output[0]], _out=_data.output)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class ReduceSum_To_ELTS3_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    transpose_1 = onnx_make_dummy_node('onnx/transpose_1', 'Transpose')
    transpose_2 = onnx_make_dummy_node('onnx/transpose_2', 'Transpose')
    unsqueeze_1 = onnx_make_dummy_node('onnx/unsqueeze_1', 'Unsqueeze')
    unsqueeze_2 = onnx_make_dummy_node('onnx/unsqueeze_2', 'Unsqueeze')
    concat = onnx_make_dummy_node('onnx/concat', 'Concat')
    reducesum = onnx_make_dummy_node('onnx/reducesum', 'ReduceSum')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    transpose_3 = onnx_make_dummy_node('onnx/transpose_3', 'Transpose')
    transpose_4 = onnx_make_dummy_node('onnx/transpose_4', 'Transpose')
    unsqueeze_3 = onnx_make_dummy_node('onnx/unsqueeze_3', 'Unsqueeze')
    for node in [transpose_1, transpose_2, unsqueeze_1, unsqueeze_2, concat, reducesum, mul, transpose_3, transpose_4, unsqueeze_3]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/transpose_1', 'onnx/unsqueeze_1')))
    edge_list.append(tuple(('onnx/transpose_2', 'onnx/unsqueeze_2')))
    edge_list.append(tuple(('onnx/unsqueeze_1', 'onnx/concat')))
    edge_list.append(tuple(('onnx/unsqueeze_2', 'onnx/concat')))
    edge_list.append(tuple(('onnx/concat', 'onnx/reducesum')))
    edge_list.append(tuple(('onnx/reducesum', 'onnx/mul')))
    edge_list.append(tuple(('onnx/mul', 'onnx/transpose_3')))
    edge_list.append(tuple(('onnx/transpose_4', 'onnx/unsqueeze_3')))
    edge_list.append(tuple(('onnx/unsqueeze_3', 'onnx/concat')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/transpose_1', 'onnx/transpose_2', 'onnx/transpose_4']

  def output_node(self):
    return "onnx/transpose_3"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "ReduceSum_To_ELTS3", _type=self.value_info[_data.output[0]], _out=_data.output)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class ReduceSum_To_ELTS2_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    transpose_1 = onnx_make_dummy_node('onnx/transpose_1', 'Transpose')
    transpose_2 = onnx_make_dummy_node('onnx/transpose_2', 'Transpose')
    unsqueeze_1 = onnx_make_dummy_node('onnx/unsqueeze_1', 'Unsqueeze')
    unsqueeze_2 = onnx_make_dummy_node('onnx/unsqueeze_2', 'Unsqueeze')
    concat = onnx_make_dummy_node('onnx/concat', 'Concat')
    reducesum = onnx_make_dummy_node('onnx/reducesum', 'ReduceSum')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    transpose_3 = onnx_make_dummy_node('onnx/transpose_3', 'Transpose')
    for node in [transpose_1, transpose_2, unsqueeze_1, unsqueeze_2, concat, reducesum, mul, transpose_3]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/transpose_1', 'onnx/unsqueeze_1')))
    edge_list.append(tuple(('onnx/transpose_2', 'onnx/unsqueeze_2')))
    edge_list.append(tuple(('onnx/unsqueeze_1', 'onnx/concat')))
    edge_list.append(tuple(('onnx/unsqueeze_2', 'onnx/concat')))
    edge_list.append(tuple(('onnx/concat', 'onnx/reducesum')))
    edge_list.append(tuple(('onnx/reducesum', 'onnx/mul')))
    edge_list.append(tuple(('onnx/mul', 'onnx/transpose_3')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/transpose_1', 'onnx/transpose_2']

  def output_node(self):
    return "onnx/transpose_3"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "ReduceSum_To_ELTS2", _type=self.value_info[_data.output[0]], _out=_data.output)
    mul_initializer = source_graph.nodes[matched_subgraph['onnx/mul']]['data'].input[1]
    node_def.input[:] = inputs_to_collapsed + [mul_initializer]
    return node_def


class Centernet_IR4_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    shape1 = onnx_make_dummy_node('onnx/shape_1', 'Shape')
    gather1 = onnx_make_dummy_node('onnx/gather_1', 'Gather')
    cast1 = onnx_make_dummy_node('onnx/cast_1', 'Cast')
    slice1 = onnx_make_dummy_node('onnx/slice_1', 'Slice')
    concat1 = onnx_make_dummy_node('onnx/concat_1', 'Concat')
    cast2 = onnx_make_dummy_node('onnx/cast_2', 'Cast')
    shape2 = onnx_make_dummy_node('onnx/shape_2', 'Shape')
    gather2 = onnx_make_dummy_node('onnx/gather_2', 'Gather')
    slice2 = onnx_make_dummy_node('onnx/slice_2', 'Slice')
    slice3 = onnx_make_dummy_node('onnx/slice_3', 'Slice')
    slice4 = onnx_make_dummy_node('onnx/slice_4', 'Slice')
    slice5 = onnx_make_dummy_node('onnx/slice_5', 'Slice')
    sub1 = onnx_make_dummy_node('onnx/sub_1', 'Sub')
    sub2 = onnx_make_dummy_node('onnx/sub_2', 'Sub')
    div1 = onnx_make_dummy_node('onnx/div_1', 'Div')
    div2 = onnx_make_dummy_node('onnx/div_2', 'Div')
    add1 = onnx_make_dummy_node('onnx/add_1', 'Add')
    add2 = onnx_make_dummy_node('onnx/add_2', 'Add')
    concat2 = onnx_make_dummy_node('onnx/concat_2', 'Concat')
    concat3 = onnx_make_dummy_node('onnx/concat_3', 'Concat')
    slice6 = onnx_make_dummy_node('onnx/slice_6', 'Slice')
    transpose1 = onnx_make_dummy_node('onnx/transpose_1', 'Transpose')
    add3 = onnx_make_dummy_node('onnx/add_3', 'Add')
    transpose2 = onnx_make_dummy_node('onnx/transpose_2', 'Transpose')
    for node in [shape1, gather1, cast1, slice1, concat1, cast2, shape2, gather2, slice2, slice3, slice4, slice5, sub1, sub2, div1, div2, add1, add2, concat2, concat3, slice6, transpose1, add3, transpose2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/shape_1', 'onnx/gather_1')))
    edge_list.append(tuple(('onnx/gather_1', 'onnx/cast_1')))
    edge_list.append(tuple(('onnx/cast_1', 'onnx/slice_1')))
    edge_list.append(tuple(('onnx/slice_1', 'onnx/concat_1')))
    edge_list.append(tuple(('onnx/concat_1', 'onnx/cast_2')))
    edge_list.append(tuple(('onnx/shape_2', 'onnx/gather_2')))
    edge_list.append(tuple(('onnx/gather_2', 'onnx/slice_2')))
    edge_list.append(tuple(('onnx/gather_2', 'onnx/slice_3')))
    edge_list.append(tuple(('onnx/cast_2', 'onnx/slice_4')))
    edge_list.append(tuple(('onnx/cast_2', 'onnx/slice_5')))
    edge_list.append(tuple(('onnx/slice_2', 'onnx/sub_1')))
    edge_list.append(tuple(('onnx/slice_4', 'onnx/sub_1')))
    edge_list.append(tuple(('onnx/slice_3', 'onnx/sub_2')))
    edge_list.append(tuple(('onnx/slice_5', 'onnx/sub_2')))
    edge_list.append(tuple(('onnx/sub_1', 'onnx/div_1')))
    edge_list.append(tuple(('onnx/sub_2', 'onnx/div_2')))
    edge_list.append(tuple(('onnx/div_1', 'onnx/concat_2')))
    edge_list.append(tuple(('onnx/div_1', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/slice_4', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/div_2', 'onnx/concat_2')))
    edge_list.append(tuple(('onnx/div_2', 'onnx/add_2')))
    edge_list.append(tuple(('onnx/slice_5', 'onnx/add_2')))
    edge_list.append(tuple(('onnx/add_1', 'onnx/concat_3')))
    edge_list.append(tuple(('onnx/add_2', 'onnx/concat_3')))
    edge_list.append(tuple(('onnx/concat_2', 'onnx/slice_6')))
    edge_list.append(tuple(('onnx/concat_3', 'onnx/slice_6')))
    edge_list.append(tuple(('onnx/transpose_1', 'onnx/slice_6')))
    edge_list.append(tuple(('onnx/slice_6', 'onnx/add_3')))
    edge_list.append(tuple(('onnx/add_3', 'onnx/transpose_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/transpose_1', 'onnx/shape_1', 'onnx/shape_2']

  def output_node(self):
    return "onnx/transpose_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing IR4")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Centernet_IR4", _type=self.value_info[_data.output[0]], _out=_data.output)
    #shape = self.state_dict
    starts = source_graph.nodes[matched_subgraph['onnx/concat_2']]['data'].input
    ends = source_graph.nodes[matched_subgraph['onnx/concat_3']]['data'].input
    node_def.input[:] = inputs_to_collapsed
    return node_def


class Swish_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    sigmoid1 = onnx_make_dummy_node('onnx/sigmoid_1', 'Sigmoid')
    mul1 = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    for node in [sigmoid1, mul1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/sigmoid_1', 'onnx/mul_1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/sigmoid_1']

  def output_node(self):
    return "onnx/mul_1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing Swish")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Swish", _type=self.value_info[_data.output[0]], _out=_data.output)
    inputs_to_sig = source_graph.nodes[matched_subgraph['onnx/sigmoid_1']]['data'].input
    inputs_to_mul = source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input
    if len(set(inputs_to_sig).intersection(set(inputs_to_mul))) == 0:
      return None
    node_def.input[:] = inputs_to_collapsed
    return node_def


class Mish_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    softplus1 = onnx_make_dummy_node('onnx/softplus_1', 'Softplus')
    tanh1 = onnx_make_dummy_node('onnx/tanh_1', 'Tanh')
    mul1 = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    for node in [softplus1, tanh1, mul1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/softplus_1', 'onnx/tanh_1')))
    edge_list.append(tuple(('onnx/tanh_1', 'onnx/mul_1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/softplus_1']

  def output_node(self):
    return "onnx/mul_1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing Mish")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Mish")
    inputs_to_sof = source_graph.nodes[matched_subgraph['onnx/softplus_1']]['data'].input
    inputs_to_mul = source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input
    if len(set(inputs_to_sof).intersection(set(inputs_to_mul))) == 0:
      return None
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class Yolov8_scatterND_Replacer(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict
    self.replacement_output_node = None
    self.replacement_is_graph = True
    self.input_node_to_replacement_graph_dict = dict()
    self.counter = 0

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    slice1 = onnx_make_dummy_node('onnx/slice_1', 'Slice')
    sigmoid = onnx_make_dummy_node('onnx/sigmoid', 'Sigmoid')
    scatternd1 = onnx_make_dummy_node('onnx/scatterND_1', 'ScatterND')
    slice2 = onnx_make_dummy_node('onnx/slice_2', 'Slice')
    mul1 = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    add1 = onnx_make_dummy_node('onnx/add_1', 'Add')
    mul2 = onnx_make_dummy_node('onnx/mul_2', 'Mul')
    scatternd2 = onnx_make_dummy_node('onnx/scatterND_2', 'ScatterND')
    slice3 = onnx_make_dummy_node('onnx/slice_3', 'Slice')
    mul3 = onnx_make_dummy_node('onnx/mul_3', 'Mul')
    add2 = onnx_make_dummy_node('onnx/add_2', 'Add')
    mul4 = onnx_make_dummy_node('onnx/mul_4', 'Mul')
    scatternd3 = onnx_make_dummy_node('onnx/scatterND_3', 'ScatterND')
    for node in [slice1, sigmoid, scatternd1, slice2, mul1, add1, mul2, scatternd2, slice3, mul3, add2, mul4, scatternd3]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/slice_1', 'onnx/sigmoid')))
    edge_list.append(tuple(('onnx/sigmoid', 'onnx/scatterND_1')))
    edge_list.append(tuple(('onnx/scatterND_1', 'onnx/slice_2')))
    edge_list.append(tuple(('onnx/scatterND_1', 'onnx/scatterND_2')))
    edge_list.append(tuple(('onnx/slice_2', 'onnx/mul_1')))
    edge_list.append(tuple(('onnx/mul_1', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/add_1', 'onnx/mul_2')))
    edge_list.append(tuple(('onnx/mul_2', 'onnx/scatterND_2')))
    edge_list.append(tuple(('onnx/scatterND_2', 'onnx/slice_3')))
    edge_list.append(tuple(('onnx/scatterND_2', 'onnx/scatterND_3')))
    edge_list.append(tuple(('onnx/slice_3', 'onnx/mul_3')))
    edge_list.append(tuple(('onnx/mul_3', 'onnx/add_2')))
    edge_list.append(tuple(('onnx/add_2', 'onnx/mul_4')))
    edge_list.append(tuple(('onnx/mul_4', 'onnx/scatterND_3')))
    G.add_edges_from(edge_list)
    return G

  def subgraph_to_replace(self, source_graph, matched_subgraph) -> nx.DiGraph:
    """Returns replacement subgraph template"""
    G = nx.DiGraph()
    edge_list = list()
    self.reshape1 = 'custom_reshape_1'
    self.slice1 = 'custom_slice_1'
    self.slice2 = 'custom_slice_2'
    self.mul1 = 'custom_mul_1'
    self.add1 = 'custom_add_1'
    self.mul2 = 'custom_mul_2'
    self.sigmoid = 'custom_sigmoid_1'
    self.concat = 'custom_concat_1'
    self.reshape2 = 'custom_reshape_2'
    reshape1_node = onnx_make_dummy_node(self.reshape1, 'Reshape')
    slice1_node = onnx_make_dummy_node(self.slice1, 'Slice')
    slice2_node = onnx_make_dummy_node(self.slice2, 'Slice')
    mul1_node = onnx_make_dummy_node(self.mul1, 'Mul')
    add1_node = onnx_make_dummy_node(self.add1, 'Add')
    mul2_node = onnx_make_dummy_node(self.mul2, 'Mul')
    sigmoid_node = onnx_make_dummy_node(self.sigmoid, 'Sigmoid')
    concat_node = onnx_make_dummy_node(self.concat, 'Concat')
    reshape2_node = onnx_make_dummy_node(self.reshape2, 'Reshape', _out=source_graph.nodes[matched_subgraph['onnx/scatterND_3']]['data'].output)
    for node in [reshape1_node, slice1_node, slice2_node, mul1_node, add1_node, mul2_node, sigmoid_node, concat_node, reshape2_node]:
      G.add_node(node.name, data=node, type=node.op_type)
    # Follow DFS (left first then right) so that inputs to nodes are being added in the correct order in mk_replacement_graph
    edge_list.append(tuple((self.reshape1, self.slice1)))
    edge_list.append(tuple((self.slice1, self.mul1)))
    edge_list.append(tuple((self.mul1, self.add1)))
    edge_list.append(tuple((self.add1, self.mul2)))
    edge_list.append(tuple((self.mul2, self.concat)))
    edge_list.append(tuple((self.concat, self.reshape2)))
    edge_list.append(tuple((self.reshape1, self.slice2)))
    edge_list.append(tuple((self.slice2, self.sigmoid)))
    edge_list.append(tuple((self.sigmoid, self.concat)))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    return ['onnx/scatterND_1', 'onnx/slice_1']

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    return "onnx/scatterND_3"

  def get_replacement_output_node(self) -> str:
    """Returns the name of the output node in the replacement graph"""
    return self.replacement_output_node

  def get_input_to_replacement_graph_dict(self):
    """"Returns dictionary mapping node in replacement graph to corresponding node outside the replacement graph feeding input"""
    return self.input_node_to_replacement_graph_dict

  def mk_replacement_graph(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None) -> nx.DiGraph:
    """
    Constructs a replacement node
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: The matched subgraph
      model: ONNX Model
      inputs_to_collaphse:
    Returns:
      replacement_graph: The netx subgraph to act as the replacement
    """

    # checking whether slice_1 and scatterND_1 has the same input (concat)
    if (list(source_graph.predecessors(matched_subgraph['onnx/slice_1']))[0] != list(source_graph.predecessors(matched_subgraph['onnx/scatterND_1']))[0]):
      return None

    logger.info("replacing Yolov8 pose scatterND pattern")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    replacement_graph = self.subgraph_to_replace(source_graph, matched_subgraph)
    self.reshape2 = matched_subgraph[self.output_node()]

    # Using a name mapping to replace placeholder node names with what we need
    name_mapping = {'custom_reshape_2': self.reshape2}
    idx = self.update_counter()
    if idx > 0:  #In case there are multiple matches
      [self.reshape1, self.slice1, self.slice2, self.mul1, self.add1, self.mul2, self.sigmoid, self.concat] = rename_layers([self.reshape1, self.slice1, self.slice2, self.mul1, self.add1, self.mul2, self.sigmoid, self.concat], name_mapping, idx)
    # Replacing names inside graph and graph data
    replacement_graph = relabel_netx_graph(replacement_graph, name_mapping)

    # Setting output node of replacement graph
    self.replacement_output_node = self.reshape2
    # Mapping nodes in replacement graph to inputs from outside the replacement graph
    self.input_node_to_replacement_graph_dict = {self.reshape1: list(source_graph.predecessors(matched_subgraph['onnx/slice_1']))[0]}

    input_shape = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/slice_1']]['data'].input[0]]
    if (input_shape[1] % 3 != 0):
      return None
    new_shape = [input_shape[0], int(input_shape[1] / 3), 3, input_shape[2]]  #[1,17,3,8400]
    #Setting output nodes
    # for node in list(replacement_graph.nodes):
    #   replacement_graph.nodes[node]['data'].output[:] = [f"{replacement_graph.nodes[node]['data'].name}_output"]
    #Setting predecessor input nodes
    for node in replacement_graph.nodes:
      for predecessor in replacement_graph.predecessors(node):
        replacement_graph.nodes[node]['data'].input.extend([replacement_graph.nodes[predecessor]['data'].output[0]])
    #Getting bias node names that can be reused in the replacement graph
    tensor_name_list = [source_graph.nodes[x]['data'].output[0] for x in matched_subgraph.values()]

    def get_init_node(node):
      node_inputs = set(source_graph.nodes[node]['data'].input)
      return [i for i in node_inputs if i not in tensor_name_list][0]

    mul1_bias = get_init_node(matched_subgraph['onnx/mul_1'])
    mul2_bias = get_init_node(matched_subgraph['onnx/mul_2'])

    #Setting input from outside to replacement graph and tensor input of reshape node
    replacement_graph.nodes[self.reshape1]['data'].input[:] = [source_graph.nodes[matched_subgraph['onnx/slice_1']]['data'].input[0], f'{self.reshape1}_shape']
    set_dict(self.state_dict, f'{self.reshape1}_shape', np.array(new_shape[:-1] + [-1]))
    set_dict(self.shape_dict, replacement_graph.nodes[self.reshape1]['data'].output[0], new_shape)

    #Updating tensor inputs to slice1
    slice_input_suffixes = ['starts', 'ends', 'axes', 'steps']
    replacement_graph.nodes[self.slice1]['data'].input.extend([f'{self.slice1}_starts', f'{self.slice1}_ends', f'{self.slice1}_axes', f'{self.slice1}_steps'])
    set_dict_iterative(self.state_dict, [f'{self.slice1}_{suffix}' for suffix in slice_input_suffixes], [np.array([num]) for num in [0, 2, 2, 1]])
    set_dict(self.shape_dict, replacement_graph.nodes[self.slice1]['data'].output[0], [new_shape[0], new_shape[1], 2, new_shape[3]])

    #Updating tensor inputs to mul1
    replacement_graph.nodes[self.mul1]['data'].input.extend([mul1_bias])
    set_dict(self.shape_dict, replacement_graph.nodes[self.mul1]['data'].output[0], [new_shape[0], new_shape[1], 2, new_shape[3]])

    #Updating tensor inputs to add1
    replacement_graph.nodes[self.add1]['data'].input.extend([f'{self.add1}_B'])
    set_dict(self.state_dict, f'{self.add1}_B', np.vstack((self.state_dict[source_graph.nodes[matched_subgraph['onnx/add_1']]['data'].input[1]], self.state_dict[source_graph.nodes[matched_subgraph['onnx/add_2']]['data'].input[1]])))
    set_dict(self.shape_dict, replacement_graph.nodes[self.add1]['data'].output[0], [new_shape[0], new_shape[1], 2, new_shape[3]])

    #Updating tensor inputs to mul2
    replacement_graph.nodes[self.mul2]['data'].input.extend([mul2_bias])
    set_dict(self.shape_dict, replacement_graph.nodes[self.mul2]['data'].output[0], [new_shape[0], new_shape[1], 2, new_shape[3]])

    #Updating tensor inputs to slice2
    replacement_graph.nodes[self.slice2]['data'].input.extend([f'{self.slice2}_starts', f'{self.slice2}_ends', f'{self.slice2}_axes', f'{self.slice2}_steps'])
    set_dict_iterative(self.state_dict, [f'{self.slice2}_{suffix}' for suffix in slice_input_suffixes], [np.array([num]) for num in [2, 3, 2, 1]])
    set_dict(self.shape_dict, replacement_graph.nodes[self.slice2]['data'].output[0], [new_shape[0], new_shape[1], 1, new_shape[3]])

    #Updating tensor inputs to sigmoid layer
    set_dict(self.shape_dict, replacement_graph.nodes[self.sigmoid]['data'].output[0], [new_shape[0], new_shape[1], 1, new_shape[3]])

    #Adding axis attribute to concat layer
    add_attribute(replacement_graph.nodes[self.concat]['data'], "axis", 2)
    set_dict(self.shape_dict, replacement_graph.nodes[self.concat]['data'].output[0], new_shape)

    #Updating tensor input to reshape2 layer
    replacement_graph.nodes[self.reshape2]['data'].input.extend([f'{self.reshape2}_shape'])
    set_dict(self.state_dict, f'{self.reshape2}_shape', np.array([input_shape[0], input_shape[1], -1]))
    set_dict(self.shape_dict, replacement_graph.nodes[self.reshape2]['data'].output[0], input_shape)
    return replacement_graph


class TestGraphReplacer(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.replacement_output_node = None
    self.replacement_is_graph = True
    self.input_node_to_replacement_graph_dict = dict()

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    softplus1 = onnx_make_dummy_node('onnx/softplus_1', 'Softplus')
    tanh1 = onnx_make_dummy_node('onnx/tanh_1', 'Tanh')
    for node in [softplus1, tanh1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/softplus_1', 'onnx/tanh_1')))
    G.add_edges_from(edge_list)
    return G

  def subgraph_to_replace(self) -> nx.DiGraph:
    """Returns replacement subgraph template"""
    G = nx.DiGraph()
    edge_list = list()
    tanh = onnx_make_dummy_node('placeholder_tanh', 'Tanh')
    softplus = onnx_make_dummy_node('placeholder_softplus', 'Softplus')
    for node in [tanh, softplus]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('placeholder_tanh', 'placeholder_softplus')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    return ['onnx/softplus_1']

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    return "onnx/tanh_1"

  def get_replacement_output_node(self) -> str:
    """Returns the name of the output node in the replacement graph"""
    return self.replacement_output_node

  def get_input_to_replacement_graph_dict(self):
    """"Returns dictionary mapping node in replacement graph to corresponding node outside the replacement graph feeding input"""
    return self.input_node_to_replacement_graph_dict

  def mk_replacement_graph(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None) -> nx.DiGraph:
    """
    Constructs a replacement node
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: The matched subgraph
      model: ONNX Model
      inputs_to_collaphse:
    Returns:
      replacement_graph: The netx subgraph to act as the replacement
    """
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    replacement_graph = self.subgraph_to_replace()
    new_tanh = matched_subgraph['onnx/tanh_1'] + '_new'
    new_softplus = matched_subgraph['onnx/softplus_1'] + '_new'
    name_mapping = {'placeholder_tanh': new_tanh, 'placeholder_softplus': new_softplus}
    replacement_graph = nx.relabel_nodes(replacement_graph, name_mapping)
    replacement_graph.nodes[new_tanh]['data'].name = new_tanh
    replacement_graph.nodes[new_tanh]['data'].input[:] = inputs_to_collapsed
    replacement_graph.nodes[new_tanh]['data'].output[:] = [new_tanh]
    replacement_graph.nodes[new_softplus]['data'].name = new_softplus
    replacement_graph.nodes[new_softplus]['data'].input[:] = [new_tanh]
    replacement_graph.nodes[new_softplus]['data'].output[:] = [new_softplus]
    #Set output node
    self.replacement_output_node = new_softplus
    self.input_node_to_replacement_graph_dict = {new_tanh: list(source_graph.predecessors(matched_subgraph['onnx/softplus_1']))[0]}
    print(f"Data output: {_data.output}")
    print(f"_type is : {self.value_info[_data.output[0]]}")
    return replacement_graph


class TestConcat_Matcher(BaseMatcher):

  def __init__(self, model, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.shape_dict = shape_dict
    self.counter = 0
    self.replacement_is_graph = True
    self.replacement_output_node = None

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    concat1 = onnx_make_dummy_node('onnx/concat', 'Concat')
    for node in [concat1]:
      G.add_node(node.name, data=node, type=node.op_type)
    return G

  def subgraph_to_replace(self) -> nx.DiGraph:
    """Returns replacement subgraph template"""
    G = nx.DiGraph()
    edge_list = list()
    self.new_concat1 = 'concat_ara1'
    self.new_concat2 = 'concat_ara2'
    self.new_concat3 = 'concat_ara3'
    concat1_node = onnx_make_dummy_node(self.new_concat1, 'Concat')
    concat2_node = onnx_make_dummy_node(self.new_concat2, 'Concat')
    concat3_node = onnx_make_dummy_node(self.new_concat3, 'Concat')
    for node in [concat1_node, concat2_node, concat3_node]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple((self.new_concat1, self.new_concat3)))
    edge_list.append(tuple((self.new_concat2, self.new_concat3)))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    return ['onnx/concat']

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    return "onnx/concat"

  def get_replacement_output_node(self) -> str:
    """Returns the name of the output node in the replacement graph"""
    return self.replacement_output_node

  def get_input_to_replacement_graph_dict(self):
    """"Returns dictionary mapping node in replacement graph to corresponding node outside the replacement graph feeding input"""
    return self.input_node_to_replacement_graph_dict

  def mk_replacement_graph(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None) -> nx.DiGraph:
    """
    Constructs a replacement node
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: The matched subgraph
      model: ONNX Model
      inputs_to_collaphse: 
    Returns:
      replacement_graph: Replacement netx subgraph
    """
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    _axis = _data.attribute[0].i
    if (len(_data.input) != 4):
      return None
    replacement_graph = self.subgraph_to_replace()
    # Reusing source node name for output of replacement node
    self.new_concat3 = matched_subgraph['onnx/concat']
    name_mapping = {'concat_ara3': self.new_concat3}
    idx = self.update_counter()
    if idx > 0:  #In case there are multiple matches, pass all the layer names that would name renaming
      [self.new_concat1, self.new_concat2] = rename_layers([self.new_concat1, self.new_concat2], name_mapping, idx)
    # Replacing names inside graph and graph data
    replacement_graph = relabel_netx_graph(replacement_graph, name_mapping)

    # Setting inputs, outputs and tensor shapes
    replacement_graph.nodes[self.new_concat1]['data'].input[:] = _data.input[:2]
    replacement_graph.nodes[self.new_concat1]['data'].output[:] = [self.new_concat1]
    add_attribute(replacement_graph.nodes[self.new_concat1]['data'], "axis", _axis)
    old_shape = deepcopy(self.shape_dict[_data.input[0]])
    old_shape[_axis] *= 2
    new_shape_1 = deepcopy(old_shape)
    set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat1]['data'].output[0], new_shape_1)

    replacement_graph.nodes[self.new_concat2]['data'].input[:] = _data.input[2:4]
    replacement_graph.nodes[self.new_concat2]['data'].output[:] = [self.new_concat2]
    add_attribute(replacement_graph.nodes[self.new_concat2]['data'], "axis", _axis)
    set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat2]['data'].output[0], new_shape_1)

    replacement_graph.nodes[self.new_concat3]['data'].input[:] = [self.new_concat1, self.new_concat2]
    replacement_graph.nodes[self.new_concat3]['data'].output[:] = [self.new_concat3]
    add_attribute(replacement_graph.nodes[self.new_concat3]['data'], "axis", _axis)
    old_shape[_axis] *= 2
    new_shape_2 = deepcopy(old_shape)
    set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat3]['data'].output[0], new_shape_2)

    #Set output node
    self.replacement_output_node = self.new_concat3
    self.input_node_to_replacement_graph_dict = {self.new_concat1: list(source_graph.predecessors(matched_subgraph['onnx/concat']))[0], self.new_concat2: list(source_graph.predecessors(matched_subgraph['onnx/concat']))[0]}
    return replacement_graph


class Yolo4_Expand_To_Upsample_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    reshape_1 = onnx_make_dummy_node('onnx/reshape_1', 'Reshape')
    expand_1 = onnx_make_dummy_node('onnx/expand_1', 'Expand')
    reshape_2 = onnx_make_dummy_node('onnx/reshape_2', 'Reshape')
    for node in [reshape_1, expand_1, reshape_2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reshape_1', 'onnx/expand_1')))
    edge_list.append(tuple(('onnx/expand_1', 'onnx/reshape_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reshape_1']

  def output_node(self):
    return "onnx/reshape_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing Reshape Expand Reshape in Yolo4 to Upsample")
    for node_number in source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data'].input:
      if node_number in self.value_info:
        output_name_reshape1 = node_number
        break
    output_name_reshape2 = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    reshape1_shape = [dim.dim_value for dim in self.value_info[output_name_reshape1].type.tensor_type.shape.dim]
    reshape2_shape = [dim.dim_value for dim in self.value_info[output_name_reshape2].type.tensor_type.shape.dim]
    check = True
    for idx in range(2, len(reshape1_shape)):
      if reshape2_shape[idx] != 2 * reshape1_shape[idx]:
        check = False
        break
    if check == False:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Upsample", scale_factor=2, mode="nearest")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class L2NormMatcher_clip(BaseMatcher):

  def __init__(self, model):
    self.replacement_is_subgraph = False

  def subgraph_to_match(self):
    G = nx.DiGraph()
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    reducesum = onnx_make_dummy_node('onnx/reducesum', 'ReduceSum')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [mul, reducesum, sqrt, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/mul', 'onnx/reducesum')))
    edge_list.append(tuple(('onnx/reducesum', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul']

  def output_node(self):
    return 'onnx/div'

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove):
    input_data = source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data']
    if (input_data.input[0] != input_data.input[1]):
      return None
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    if (input_data.input[0] != _data.input[0]):
      return None
    logger.info("Replacing L2Norm Clip")
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm")
    node_def.attribute.append(helper.make_attribute("across_spatial", False))
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = _data.output
    return node_def


class L2NormMatcher_simba(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    l2_reduce = onnx_make_dummy_node('onnx/reduceL2', 'ReduceL2')
    shape = onnx_make_dummy_node('onnx/shape', 'Shape')
    exp = onnx_make_dummy_node('onnx/expand', 'Expand')
    clip = onnx_make_dummy_node('onnx/clip', 'Clip')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [l2_reduce, shape, exp, clip, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reduceL2', 'onnx/clip')))
    edge_list.append(tuple(('onnx/clip', 'onnx/expand')))
    edge_list.append(tuple(('onnx/shape', 'onnx/expand')))
    edge_list.append(tuple(('onnx/expand', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reduceL2']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm")
    node_def.input[:] = inputs_to_collapsed
    return node_def


class L2NormMatcher_simba_1(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    l2_reduce = onnx_make_dummy_node('onnx/reduceL2', 'ReduceL2')
    exp = onnx_make_dummy_node('onnx/expand', 'Expand')
    clip = onnx_make_dummy_node('onnx/clip', 'Clip')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [l2_reduce, exp, clip, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reduceL2', 'onnx/clip')))
    edge_list.append(tuple(('onnx/clip', 'onnx/expand')))
    edge_list.append(tuple(('onnx/expand', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reduceL2']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = _data.output
    return node_def


class L2NormMatcher_1(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    l2_reduce = onnx_make_dummy_node('onnx/reduceL2', 'ReduceL2')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [l2_reduce, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reduceL2', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reduceL2']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    inputs_to_red = source_graph.nodes[matched_subgraph['onnx/reduceL2']]['data'].input
    inputs_to_div = source_graph.nodes[matched_subgraph['onnx/div']]['data'].input
    if len(set(inputs_to_red).intersection(set(inputs_to_div))) == 0:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "ReduceL2")
    node_def.input[:] = inputs_to_collapsed
    _proto_copy_attribute(source_graph.nodes[matched_subgraph['onnx/reduceL2']]['data'], node_def)
    node_def.output[:] = _data.output
    return node_def


class Convnext_simplify_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/Transpose_1', 'Transpose')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ReduceMean_1', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Pow', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ReduceMean_2', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_1', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sqrt', 'Sqrt')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_1', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_1', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_2', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/MatMul_1', 'MatMul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_3', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_2', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Erf', 'Erf')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_4', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_2', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_3', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/MatMul_2', 'MatMul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_5', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_4', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Transpose_2', 'Transpose')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/Transpose_1', 'onnx/ReduceMean_1')))
    edge_list.append(tuple(('onnx/Transpose_1', 'onnx/Sub')))
    edge_list.append(tuple(('onnx/ReduceMean_1', 'onnx/Sub')))
    edge_list.append(tuple(('onnx/Sub', 'onnx/Pow')))
    edge_list.append(tuple(('onnx/Pow', 'onnx/ReduceMean_2')))
    edge_list.append(tuple(('onnx/ReduceMean_2', 'onnx/Add_1')))
    edge_list.append(tuple(('onnx/Add_1', 'onnx/Sqrt')))
    edge_list.append(tuple(('onnx/Sqrt', 'onnx/Div_1')))
    edge_list.append(tuple(('onnx/Sub', 'onnx/Div_1')))
    edge_list.append(tuple(('onnx/Div_1', 'onnx/Mul_1')))
    edge_list.append(tuple(('onnx/Mul_1', 'onnx/Add_2')))
    edge_list.append(tuple(('onnx/Add_2', 'onnx/MatMul_1')))
    edge_list.append(tuple(('onnx/MatMul_1', 'onnx/Add_3')))
    edge_list.append(tuple(('onnx/Add_3', 'onnx/Div_2')))
    edge_list.append(tuple(('onnx/Div_2', 'onnx/Erf')))
    edge_list.append(tuple(('onnx/Erf', 'onnx/Add_4')))
    edge_list.append(tuple(('onnx/Add_4', 'onnx/Mul_2')))
    edge_list.append(tuple(('onnx/Add_3', 'onnx/Mul_2')))
    edge_list.append(tuple(('onnx/Mul_2', 'onnx/Mul_3')))
    edge_list.append(tuple(('onnx/Mul_3', 'onnx/MatMul_2')))
    edge_list.append(tuple(('onnx/MatMul_2', 'onnx/Add_5')))
    edge_list.append(tuple(('onnx/Add_5', 'onnx/Mul_4')))
    edge_list.append(tuple(('onnx/Mul_4', 'onnx/Transpose_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Transpose_1']

  def output_node(self):
    return "onnx/Transpose_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # names
    first_name = matched_subgraph[self.input_nodes()[0]]
    conv_name_1 = matched_subgraph['onnx/MatMul_1']
    conv_name_2 = matched_subgraph['onnx/MatMul_2']
    add_3_name = matched_subgraph['onnx/Add_3']
    add_5_name = matched_subgraph['onnx/Add_5']
    last_name = matched_subgraph[self.output_node()]
    # new nodes
    lyrm_new_node = onnx_make_dummy_node(first_name, "LayerNorm")
    conv_1_new_node = onnx_make_dummy_node(conv_name_1, "Conv")
    gelu_div_node = source_graph.nodes[matched_subgraph['onnx/Div_2']]['data']
    gelu_erf_node = source_graph.nodes[matched_subgraph['onnx/Erf']]['data']
    gelu_add_node = source_graph.nodes[matched_subgraph['onnx/Add_4']]['data']
    gelu_mul_1_node = source_graph.nodes[matched_subgraph['onnx/Mul_2']]['data']
    gelu_mul_2_node = source_graph.nodes[matched_subgraph['onnx/Mul_3']]['data']
    conv_2_new_node = onnx_make_dummy_node(conv_name_2, "Conv")
    mul_4_node = source_graph.nodes[matched_subgraph['onnx/Mul_4']]['data']
    # matched nodes
    tr_node = source_graph.nodes[first_name]['data']
    add_3_node = source_graph.nodes[add_3_name]['data']
    add_5_node = source_graph.nodes[add_5_name]['data']
    last_node = source_graph.nodes[last_name]['data']
    # setting inputs/outputs
    lyrm_new_node.input[:] = tr_node.input
    lyrm_new_node.output[:] = tr_node.output
    conv_1_new_node.input[:] = tr_node.output
    conv_1_new_node.output[:] = add_3_node.output
    conv_2_new_node.input[:] = gelu_mul_2_node.output
    conv_2_new_node.output[:] = add_5_node.output
    mul_4_node.output[:] = last_node.output
    tensor_name_list = [source_graph.nodes[x]['data'].output[0] for x in matched_subgraph.values()]

    def get_init_node(node):
      node_inputs = set(source_graph.nodes[node]['data'].input)
      return [i for i in node_inputs if i not in tensor_name_list][0]

    # setting layernorm attrs
    axes_attr = next(i for i in source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data'].attribute if 'axes' == i.name)
    if axes_attr.ints != [-1]:
      return None

    eps_node = matched_subgraph['onnx/Add_1']
    scale_node = matched_subgraph['onnx/Mul_1']
    bias_node = matched_subgraph['onnx/Add_2']
    eps_init = get_init_node(eps_node)
    scale_init = get_init_node(scale_node)
    bias_init = get_init_node(bias_node)
    eps_node = self.tensor_to_name_map[eps_init]
    eps = nh.to_array(source_graph.nodes[eps_node]['data'].attribute[0].t)
    lyrm_new_node.attribute.append(helper.make_attribute("eps", np.ndarray.tolist(eps)))
    lyrm_new_node.input.append(scale_init)
    lyrm_new_node.input.append(bias_init)
    lyrm_new_node.attribute.append(helper.make_attribute("axes", 1))
    # seeting conv_1 attrs
    weight_node = matched_subgraph['onnx/MatMul_1']
    bias_node = matched_subgraph['onnx/Add_3']
    weight_init = get_init_node(weight_node)
    bias_init = get_init_node(bias_node)
    conv_1_new_node.input.extend([weight_init])
    conv_1_new_node.input.extend([bias_init])
    conv_1_new_node.attribute.append(helper.make_attribute("dilations", (1, 1)))
    conv_1_new_node.attribute.append(helper.make_attribute("group", 1))
    conv_1_new_node.attribute.append(helper.make_attribute("kernel_shape", (1, 1)))
    conv_1_new_node.attribute.append(helper.make_attribute("pads", (0, 0, 0, 0)))
    conv_1_new_node.attribute.append(helper.make_attribute("strides", (1, 1)))
    self.state_dict[conv_1_new_node.input[1]] = self.state_dict[conv_1_new_node.input[1]].transpose(1, 0)
    weight_shape = self.state_dict[conv_1_new_node.input[1]].shape
    self.state_dict[conv_1_new_node.input[1]] = self.state_dict[conv_1_new_node.input[1]].reshape(*weight_shape, 1, 1)
    # seeting conv_2 attrs
    weight_node = matched_subgraph['onnx/MatMul_2']
    bias_node = matched_subgraph['onnx/Add_5']
    weight_init = get_init_node(weight_node)
    bias_init = get_init_node(bias_node)
    conv_2_new_node.input.extend([weight_init])
    conv_2_new_node.input.extend([bias_init])
    conv_2_new_node.attribute.append(helper.make_attribute("dilations", (1, 1)))
    conv_2_new_node.attribute.append(helper.make_attribute("group", 1))
    conv_2_new_node.attribute.append(helper.make_attribute("kernel_shape", (1, 1)))
    conv_2_new_node.attribute.append(helper.make_attribute("pads", (0, 0, 0, 0)))
    conv_2_new_node.attribute.append(helper.make_attribute("strides", (1, 1)))
    self.state_dict[conv_2_new_node.input[1]] = self.state_dict[conv_2_new_node.input[1]].transpose(1, 0)
    weight_shape = self.state_dict[conv_2_new_node.input[1]].shape
    self.state_dict[conv_2_new_node.input[1]] = self.state_dict[conv_2_new_node.input[1]].reshape(*weight_shape, 1, 1)
    # modifying subgraph
    new_edge_list = []
    first_node_in = [i for i in source_graph.predecessors(first_name)]
    for i in first_node_in:
      new_edge_list.append(tuple((i, lyrm_new_node.name)))
    last_node_out = [i for i in source_graph.successors(last_name)]
    for i in last_node_out:
      new_edge_list.append(tuple((mul_4_node.name, i)))
    for i in matched_subgraph.values():
      source_graph.remove_node(i)
    new_node_list = [lyrm_new_node, conv_1_new_node, gelu_div_node, gelu_erf_node, gelu_add_node, gelu_mul_1_node, gelu_mul_2_node, conv_2_new_node, mul_4_node]
    for node in new_node_list:
      source_graph.add_node(node.name, data=node, type=node.op_type)
    for i in range(len(new_node_list) - 1):
      new_edge_list.append(tuple((new_node_list[i].name, new_node_list[i + 1].name)))
    new_edge_list.append(tuple((conv_1_new_node.name, gelu_mul_1_node.name)))
    source_graph.add_edges_from(new_edge_list)
    # fixing value_info shapes
    for i in new_node_list[:-1]:
      dim = self.value_info[i.output[0]].type.tensor_type.shape.dim
      dim[1].dim_value, dim[3].dim_value = dim[3].dim_value, dim[1].dim_value
      self.shape_dict[i.output[0]][1], self.shape_dict[i.output[0]][3] = self.shape_dict[i.output[0]][3], self.shape_dict[i.output[0]][1]
    logger.info("Convnext nhwc branch simplified " + first_name)
    return None


class LayerNormMatcher_1(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/ReduceMean_1', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub_1', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub_2', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Pow', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ReduceMean_2', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_1', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sqrt', 'Sqrt')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_2', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/ReduceMean_1', 'onnx/Sub_1')))
    edge_list.append(tuple(('onnx/ReduceMean_1', 'onnx/Sub_2')))
    edge_list.append(tuple(('onnx/Sub_1', 'onnx/Pow')))
    edge_list.append(tuple(('onnx/Pow', 'onnx/ReduceMean_2')))
    edge_list.append(tuple(('onnx/ReduceMean_2', 'onnx/Add_1')))
    edge_list.append(tuple(('onnx/Add_1', 'onnx/Sqrt')))
    edge_list.append(tuple(('onnx/Sqrt', 'onnx/Div')))
    edge_list.append(tuple(('onnx/Sub_2', 'onnx/Div')))
    edge_list.append(tuple(('onnx/Div', 'onnx/Mul')))
    edge_list.append(tuple(('onnx/Mul', 'onnx/Add_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/ReduceMean_1']

  def output_node(self):
    return "onnx/Add_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    reduce_mean_attr = source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data'].attribute
    axes_attr = next(i for i in reduce_mean_attr if 'axes' == i.name)
    axes = axes_attr.ints[0]
    tensor_name_list = [source_graph.nodes[x]['data'].output[0] for x in matched_subgraph.values()]

    def get_init_node(node):
      node_inputs = set(source_graph.nodes[node]['data'].input)
      return [i for i in node_inputs if i not in tensor_name_list][0]

    eps_node = matched_subgraph['onnx/Add_1']
    scale_node = matched_subgraph['onnx/Mul']
    bias_node = matched_subgraph['onnx/Add_2']
    eps_init = get_init_node(eps_node)
    scale_init = get_init_node(scale_node)
    bias_init = get_init_node(bias_node)
    eps_node = self.tensor_to_name_map[eps_init]
    eps = nh.to_array(source_graph.nodes[eps_node]['data'].attribute[0].t)
    node_def.attribute.append(helper.make_attribute("eps", np.ndarray.tolist(eps)))
    node_def.input.append(scale_init)
    node_def.input.append(bias_init)
    node_def.attribute.append(helper.make_attribute("axes", axes))
    return node_def


class LayerNormMatcher_2(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.init_name_map = [i.name for i in model.graph.initializer]
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/ReduceMean_1', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Pow', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ReduceMean_2', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_1', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sqrt', 'Sqrt')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_2', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/ReduceMean_1', 'onnx/Sub')))
    edge_list.append(tuple(('onnx/Sub', 'onnx/Pow')))
    edge_list.append(tuple(('onnx/Pow', 'onnx/ReduceMean_2')))
    edge_list.append(tuple(('onnx/ReduceMean_2', 'onnx/Add_1')))
    edge_list.append(tuple(('onnx/Add_1', 'onnx/Sqrt')))
    edge_list.append(tuple(('onnx/Sqrt', 'onnx/Div')))
    edge_list.append(tuple(('onnx/Sub', 'onnx/Div')))
    edge_list.append(tuple(('onnx/Div', 'onnx/Mul')))
    edge_list.append(tuple(('onnx/Mul', 'onnx/Add_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/ReduceMean_1']

  def output_node(self):
    return "onnx/Add_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    reduce_1_inputs = set(source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data'].input)
    reduce_1_outputs = set(source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data'].output)
    if matched_subgraph['onnx/Sub'] not in source_graph.nodes:
      return None
    sub_inputs = set(source_graph.nodes[matched_subgraph['onnx/Sub']]['data'].input)
    if reduce_1_outputs.union(reduce_1_inputs) != sub_inputs:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    reduce_mean_attr = source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data'].attribute
    axes_attr = next(i for i in reduce_mean_attr if 'axes' == i.name)
    axes = axes_attr.ints[0]
    tensor_name_list = [source_graph.nodes[x]['data'].output[0] for x in matched_subgraph.values()]

    def get_init_node(node):
      node_inputs = set(source_graph.nodes[node]['data'].input)
      return [i for i in node_inputs if i not in tensor_name_list][0]

    eps_node = matched_subgraph['onnx/Add_1']
    scale_node = matched_subgraph['onnx/Mul']
    bias_node = matched_subgraph['onnx/Add_2']
    eps_init = get_init_node(eps_node)
    scale_init = get_init_node(scale_node)
    bias_init = get_init_node(bias_node)
    eps_node = eps_init if eps_init in self.init_name_map else self.tensor_to_name_map[eps_init]
    if hasattr(source_graph.nodes[eps_node]['data'], 'attribute'):
      eps = nh.to_array(source_graph.nodes[eps_node]['data'].attribute[0].t)
    else:
      eps = nh.to_array(source_graph.nodes[eps_node]['data'])
    node_def.attribute.append(helper.make_attribute("eps", np.ndarray.tolist(eps)))
    node_def.input.append(scale_init)
    node_def.input.append(bias_init)
    node_def.attribute.append(helper.make_attribute("axes", axes))
    return node_def


class LayerNormMatcher_3(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.init_name_map = [i.name for i in model.graph.initializer]

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/GlobalAveragePool_1', 'GlobalAveragePool')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub_1', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_1', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/GlobalAveragePool_2', 'GlobalAveragePool')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_1', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sqrt', 'Sqrt')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Reciprocal', 'Reciprocal')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_2', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_3', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_4', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub_2', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_2', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/GlobalAveragePool_1', 'onnx/Sub_1')))
    edge_list.append(tuple(('onnx/Sub_1', 'onnx/Mul_1')))
    edge_list.append(tuple(('onnx/Mul_1', 'onnx/GlobalAveragePool_2')))
    edge_list.append(tuple(('onnx/GlobalAveragePool_2', 'onnx/Add_1')))
    edge_list.append(tuple(('onnx/Add_1', 'onnx/Sqrt')))
    edge_list.append(tuple(('onnx/Sqrt', 'onnx/Reciprocal')))
    edge_list.append(tuple(('onnx/Reciprocal', 'onnx/Mul_2')))
    edge_list.append(tuple(('onnx/Mul_2', 'onnx/Mul_3')))
    edge_list.append(tuple(('onnx/Mul_2', 'onnx/Mul_4')))
    edge_list.append(tuple(('onnx/GlobalAveragePool_1', 'onnx/Mul_3')))
    edge_list.append(tuple(('onnx/Mul_3', 'onnx/Sub_2')))
    edge_list.append(tuple(('onnx/Sub_2', 'onnx/Add_2')))
    edge_list.append(tuple(('onnx/Mul_4', 'onnx/Add_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/GlobalAveragePool_1']

  def output_node(self):
    return "onnx/Add_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    input = source_graph.nodes[matched_subgraph['onnx/GlobalAveragePool_1']]['data'].input[0]
    out_gap1 = source_graph.nodes[matched_subgraph['onnx/GlobalAveragePool_1']]['data'].output[0]
    inp_sub1 = source_graph.nodes[matched_subgraph['onnx/Sub_1']]['data'].input
    if inp_sub1[0] != input or inp_sub1[1] != out_gap1:
      return
    out_mul2 = source_graph.nodes[matched_subgraph['onnx/Mul_2']]['data'].output[0]
    inp_mul4 = source_graph.nodes[matched_subgraph['onnx/Mul_4']]['data'].input
    if inp_mul4[0] != input or inp_mul4[1] != out_mul2:
      return
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def.input[:] = [input]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    tensor_name_list = [source_graph.nodes[x]['data'].output[0] for x in matched_subgraph.values()]

    def get_init_node(node):
      node_inputs = set(source_graph.nodes[node]['data'].input)
      return [i for i in node_inputs if i not in tensor_name_list][0]

    eps_node = matched_subgraph['onnx/Add_1']
    scale_node = matched_subgraph['onnx/Mul_2']
    bias_node = matched_subgraph['onnx/Sub_2']
    eps_init = get_init_node(eps_node)
    scale_init = get_init_node(scale_node)
    bias_init = get_init_node(bias_node)
    eps_node = eps_init if eps_init in self.init_name_map else self.tensor_to_name_map[eps_init]
    if (eps_node in source_graph.nodes) and (hasattr(source_graph.nodes[eps_node]['data'], 'attribute')):
      eps = nh.to_array(source_graph.nodes[eps_node]['data'].attribute[0].t)
    else:
      eps = nh.to_array(source_graph.nodes[eps_node]['data'])
    node_def.attribute.append(helper.make_attribute("eps", np.ndarray.tolist(eps)))
    node_def.input.append(scale_init)
    node_def.input.append(bias_init)
    node_def.attribute.append(helper.make_attribute("axes", 2))
    return node_def


class LayerNormMatcher_4(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.init_name_map = [i.name for i in model.graph.initializer]
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/ReduceMean_1', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sub_1', 'Sub')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Pow', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ReduceMean_2', 'ReduceMean')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_1', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sqrt', 'Sqrt')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/ReduceMean_1', 'onnx/Sub_1')))
    edge_list.append(tuple(('onnx/Sub_1', 'onnx/Pow')))
    edge_list.append(tuple(('onnx/Pow', 'onnx/ReduceMean_2')))
    edge_list.append(tuple(('onnx/ReduceMean_2', 'onnx/Add_1')))
    edge_list.append(tuple(('onnx/Add_1', 'onnx/Sqrt')))
    edge_list.append(tuple(('onnx/Sqrt', 'onnx/Div')))
    edge_list.append(tuple(('onnx/Sub_1', 'onnx/Div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/ReduceMean_1']

  def output_node(self):
    return "onnx/Div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    if matched_subgraph['onnx/Sub_1'] not in source_graph.nodes:
      return None
    ReduceMean_1 = source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data']
    Sub_1 = source_graph.nodes[matched_subgraph['onnx/Sub_1']]['data']
    check = False
    if len(Sub_1.input) == 2 and len(ReduceMean_1.input) == 1 and len(ReduceMean_1.output) == 1:
      if ReduceMean_1.input[0] in Sub_1.input and ReduceMean_1.output[0] in Sub_1.input:
        check = True
    if not check:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    reduce_mean_attr = source_graph.nodes[matched_subgraph['onnx/ReduceMean_1']]['data'].attribute
    axes_attr = next(i for i in reduce_mean_attr if 'axes' == i.name)
    axes = axes_attr.ints[0]
    tensor_name_list = [source_graph.nodes[x]['data'].output[0] for x in matched_subgraph.values()]

    def get_init_node(node):
      node_inputs = set(source_graph.nodes[node]['data'].input)
      return [i for i in node_inputs if i not in tensor_name_list][0]

    eps_node = matched_subgraph['onnx/Add_1']
    # scale_node = matched_subgraph['onnx/Mul']
    # bias_node = matched_subgraph['onnx/Add_2']
    eps_init = get_init_node(eps_node)
    # scale_init = get_init_node(scale_node)
    # bias_init = get_init_node(bias_node)
    eps_node = eps_init if eps_init in self.init_name_map else self.tensor_to_name_map[eps_init]
    if hasattr(source_graph.nodes[eps_node]['data'], 'attribute'):
      eps = nh.to_array(source_graph.nodes[eps_node]['data'].attribute[0].t)
    else:
      eps = nh.to_array(source_graph.nodes[eps_node]['data'])
    # eps = nh.to_array(source_graph.nodes[eps_node]['data'].attribute[0].t)
    node_def.attribute.append(helper.make_attribute("eps", np.ndarray.tolist(eps)))
    # node_def.input.append(scale_init)
    # node_def.input.append(bias_init)
    node_def.attribute.append(helper.make_attribute("axes", axes))
    return node_def


class Advertima_IR(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    shape1 = onnx_make_dummy_node('onnx/shape_1', 'Shape')
    shape2 = onnx_make_dummy_node('onnx/shape_2', 'Shape')
    shape3 = onnx_make_dummy_node('onnx/shape_3', 'Shape')
    gather1 = onnx_make_dummy_node('onnx/gather_1', 'Gather')
    gather2 = onnx_make_dummy_node('onnx/gather_2', 'Gather')
    unsqueeze1 = onnx_make_dummy_node('onnx/unsqueeze_1', 'Unsqueeze')
    unsqueeze2 = onnx_make_dummy_node('onnx/unsqueeze_2', 'Unsqueeze')
    concat1 = onnx_make_dummy_node('onnx/concat_1', 'Concat')
    concat2 = onnx_make_dummy_node('onnx/concat_2', 'Concat')
    cast1 = onnx_make_dummy_node('onnx/cast_1', 'Cast')
    slice1 = onnx_make_dummy_node('onnx/slice_1', 'Slice')
    resize1 = onnx_make_dummy_node('onnx/resize_1', 'Resize')
    for node in [shape1, shape2, shape3, gather1, gather2, unsqueeze1, unsqueeze2, concat1, concat2, cast1, slice1, resize1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/shape_1', 'onnx/gather_1')))
    edge_list.append(tuple(('onnx/shape_2', 'onnx/gather_2')))
    edge_list.append(tuple(('onnx/gather_1', 'onnx/unsqueeze_1')))
    edge_list.append(tuple(('onnx/gather_2', 'onnx/unsqueeze_2')))
    edge_list.append(tuple(('onnx/unsqueeze_1', 'onnx/concat_1')))
    edge_list.append(tuple(('onnx/unsqueeze_2', 'onnx/concat_1')))
    edge_list.append(tuple(('onnx/concat_1', 'onnx/cast_1')))
    edge_list.append(tuple(('onnx/cast_1', 'onnx/concat_2')))
    edge_list.append(tuple(('onnx/concat_2', 'onnx/resize_1')))
    edge_list.append(tuple(('onnx/shape_3', 'onnx/slice_1')))
    edge_list.append(tuple(('onnx/slice_1', 'onnx/concat_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/shape_1', 'onnx/shape_2', 'onnx/shape_3']

  def output_node(self):
    return "onnx/resize_1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    inputs_to_collapsed.sort(reverse=True)
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Advertima_IR", _type=self.value_info[_data.output[0]], _out=_data.output)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class RoiAlignMax_Matcher(BaseMatcher):

  def __init__(self, model, state_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict

  def get_int_attribute(self, data_node: onnx.onnx_ml_pb2.NodeProto, name):
    for attr in data_node.attribute:
      if attr.name == name:
        return attr.i

  def get_ints_attribute(self, data_node: onnx.onnx_ml_pb2.NodeProto, name):
    ret = []
    for attr in data_node.attribute:
      if attr.name == name:
        ret = [i for i in attr.ints]
        return ret

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    roia = onnx_make_dummy_node('onnx/roialign', 'RoiAlign')
    mp = onnx_make_dummy_node('onnx/mp', 'MaxPool')
    for node in [roia, mp]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/roialign', 'onnx/mp')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/roialign']

  def output_node(self):
    return "onnx/mp"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing RoiAlignMax")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "RoiAlignMax", _type=self.value_info[_data.output[0]], _out=_data.output)
    roi_data = source_graph.nodes[matched_subgraph['onnx/roialign']]['data']
    mp_data = source_graph.nodes[matched_subgraph['onnx/mp']]['data']
    ks = self.get_ints_attribute(mp_data, 'kernel_shape')
    roi_s = [self.get_int_attribute(roi_data, 'output_height'), self.get_int_attribute(roi_data, 'output_width')]
    if (ks != roi_s):
      return None
    _proto_copy_attribute(roi_data, node_def)
    node_def.input[:] = inputs_to_collapsed
    nodes_to_remove.remove(node_def.name)
    return node_def


class PytorchOnnxSoftmax_PSP_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def get_ints_attribute(self, data_node: onnx.onnx_ml_pb2.NodeProto, name):
    ret = []
    for attr in data_node.attribute:
      if attr.name == name:
        ret = [i for i in attr.ints]
        return ret

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    reducemax = onnx_make_dummy_node('onnx/reducemax', 'ReduceMax')
    sub = onnx_make_dummy_node('onnx/sub', 'Sub')
    exp = onnx_make_dummy_node('onnx/exp', 'Exp')
    reducesum = onnx_make_dummy_node('onnx/reducesum', 'ReduceSum')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [reducemax, sub, exp, reducesum, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reducemax', 'onnx/sub')))
    edge_list.append(tuple(('onnx/sub', 'onnx/exp')))
    edge_list.append(tuple(('onnx/exp', 'onnx/reducesum')))
    edge_list.append(tuple(('onnx/reducesum', 'onnx/div')))
    edge_list.append(tuple(('onnx/exp', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reducemax']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing PytorchOnnxSoftmax")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "PytorchOnnxSoftmax", _type=self.value_info[_data.output[0]], _out=_data.output)
    reducemax_data = source_graph.nodes[matched_subgraph['onnx/reducemax']]['data']
    dim = self.get_ints_attribute(reducemax_data, 'axes')[0]
    node_def.input[:] = reducemax_data.input
    node_def_attr = helper.make_attribute("dim", dim)
    node_def.attribute.append(node_def_attr)
    return node_def


class PytorchOnnxSoftmax_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def get_ints_attribute(self, data_node: onnx.onnx_ml_pb2.NodeProto, name):
    ret = []
    for attr in data_node.attribute:
      if attr.name == name:
        ret = [i for i in attr.ints]
        return ret

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    exp = onnx_make_dummy_node('onnx/exp', 'Exp')
    reducesum = onnx_make_dummy_node('onnx/reducesum', 'ReduceSum')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    for node in [exp, reducesum, div]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/exp', 'onnx/reducesum')))
    edge_list.append(tuple(('onnx/reducesum', 'onnx/div')))
    edge_list.append(tuple(('onnx/exp', 'onnx/div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/exp']

  def output_node(self):
    return "onnx/div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing PytorchOnnxSoftmax")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "PytorchOnnxSoftmax", _type=self.value_info[_data.output[0]], _out=_data.output)
    reducesum_data = source_graph.nodes[matched_subgraph['onnx/reducesum']]['data']
    dim = self.get_ints_attribute(reducesum_data, 'axes')[0]
    exp_data = source_graph.nodes[matched_subgraph['onnx/exp']]['data']
    node_def.input[:] = exp_data.input
    node_def_attr = helper.make_attribute("dim", dim)
    node_def.attribute.append(node_def_attr)
    return node_def


class FocusLayer_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    slice1 = onnx_make_dummy_node('onnx/slice1', 'Slice')
    slice2 = onnx_make_dummy_node('onnx/slice2', 'Slice')
    slice3 = onnx_make_dummy_node('onnx/slice3', 'Slice')
    slice4 = onnx_make_dummy_node('onnx/slice4', 'Slice')
    slice5 = onnx_make_dummy_node('onnx/slice5', 'Slice')
    slice6 = onnx_make_dummy_node('onnx/slice6', 'Slice')
    slice7 = onnx_make_dummy_node('onnx/slice7', 'Slice')
    slice8 = onnx_make_dummy_node('onnx/slice8', 'Slice')
    concat = onnx_make_dummy_node('onnx/concat', 'Concat')
    for node in [slice1, slice2, slice3, slice4, slice5, slice6, slice7, slice8, concat]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/slice1', 'onnx/slice5')))
    edge_list.append(tuple(('onnx/slice2', 'onnx/slice6')))
    edge_list.append(tuple(('onnx/slice3', 'onnx/slice7')))
    edge_list.append(tuple(('onnx/slice4', 'onnx/slice8')))
    edge_list.append(tuple(('onnx/slice5', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice6', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice7', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice8', 'onnx/concat')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/slice1', 'onnx/slice2', 'onnx/slice3', 'onnx/slice4']

  def output_node(self):
    return "onnx/concat"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing Focus")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Focus")
    inputs_to_slice1 = set(source_graph.nodes[matched_subgraph['onnx/slice1']]['data'].input)
    inputs_to_slice2 = set(source_graph.nodes[matched_subgraph['onnx/slice2']]['data'].input)
    inputs_to_slice3 = set(source_graph.nodes[matched_subgraph['onnx/slice3']]['data'].input)
    inputs_to_slice4 = set(source_graph.nodes[matched_subgraph['onnx/slice4']]['data'].input)
    if len(set.intersection(inputs_to_slice1, inputs_to_slice2, inputs_to_slice3, inputs_to_slice4)) == 0:
      return None
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class FocusLayer2_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    slice1 = onnx_make_dummy_node('onnx/slice1', 'Slice')
    slice2 = onnx_make_dummy_node('onnx/slice2', 'Slice')
    slice3 = onnx_make_dummy_node('onnx/slice3', 'Slice')
    slice4 = onnx_make_dummy_node('onnx/slice4', 'Slice')
    slice5 = onnx_make_dummy_node('onnx/slice5', 'Slice')
    slice6 = onnx_make_dummy_node('onnx/slice6', 'Slice')
    concat = onnx_make_dummy_node('onnx/concat', 'Concat')
    for node in [slice1, slice2, slice3, slice4, slice5, slice6, concat]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/slice1', 'onnx/slice3')))
    edge_list.append(tuple(('onnx/slice1', 'onnx/slice4')))
    edge_list.append(tuple(('onnx/slice2', 'onnx/slice5')))
    edge_list.append(tuple(('onnx/slice2', 'onnx/slice6')))
    edge_list.append(tuple(('onnx/slice3', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice4', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice5', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice6', 'onnx/concat')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/slice1', 'onnx/slice2']

  def output_node(self):
    return "onnx/concat"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing Focus")
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Focus")
    inputs_to_slice1 = set(source_graph.nodes[matched_subgraph['onnx/slice1']]['data'].input)
    inputs_to_slice2 = set(source_graph.nodes[matched_subgraph['onnx/slice2']]['data'].input)
    if len(set.intersection(inputs_to_slice1, inputs_to_slice2)) == 0:
      return None
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class SFRoiMaxLayer_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    gather1 = onnx_make_dummy_node('onnx/gather1', 'Gather')
    squeeze = onnx_make_dummy_node('onnx/squeeze', 'Squeeze')
    cast = onnx_make_dummy_node('onnx/cast', 'Cast')
    # cast1 = onnx_make_dummy_node('onnx/cast1', 'Cast')
    gather = onnx_make_dummy_node('onnx/gather2', 'Gather')
    roialign = onnx_make_dummy_node('onnx/roialign', 'RoiAlignMax')
    for node in [gather1, squeeze, cast, gather, roialign]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/gather1', 'onnx/squeeze')))
    edge_list.append(tuple(('onnx/squeeze', 'onnx/cast')))
    edge_list.append(tuple(('onnx/gather2', 'onnx/roialign')))
    edge_list.append(tuple(('onnx/cast', 'onnx/roialign')))
    # edge_list.append(tuple(('onnx/cast1', 'onnx/roialign')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/gather1', 'onnx/roialign']

  def output_node(self):
    return "onnx/roialign"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing SF roi input layers")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "RoiAlignMax")
    _proto_copy_attribute(_data, node_def)
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    nodes_to_remove.remove(node_def.name)
    return node_def


class SFRoiLayer_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    gather1 = onnx_make_dummy_node('onnx/gather1', 'Gather')
    squeeze = onnx_make_dummy_node('onnx/squeeze', 'Squeeze')
    cast = onnx_make_dummy_node('onnx/cast', 'Cast')
    # cast1 = onnx_make_dummy_node('onnx/cast1', 'Cast')
    gather = onnx_make_dummy_node('onnx/gather2', 'Gather')
    roialign = onnx_make_dummy_node('onnx/roialign', 'RoiAlign')
    for node in [gather1, squeeze, cast, gather, roialign]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/gather1', 'onnx/squeeze')))
    edge_list.append(tuple(('onnx/squeeze', 'onnx/cast')))
    edge_list.append(tuple(('onnx/gather2', 'onnx/roialign')))
    edge_list.append(tuple(('onnx/cast', 'onnx/roialign')))
    # edge_list.append(tuple(('onnx/cast1', 'onnx/roialign')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/gather1', 'onnx/roialign']

  def output_node(self):
    return "onnx/roialign"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing SF roi input layers")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "RoiAlign")
    _proto_copy_attribute(_data, node_def)
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    nodes_to_remove.remove(node_def.name)
    return node_def


class SFRoi2Layer_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    gather1 = onnx_make_dummy_node('onnx/gather1', 'Gather')
    squeeze = onnx_make_dummy_node('onnx/squeeze', 'Squeeze')
    cast = onnx_make_dummy_node('onnx/cast', 'Cast')
    sub = onnx_make_dummy_node('onnx/sub', 'Sub')
    gather = onnx_make_dummy_node('onnx/gather2', 'Gather')
    roialign = onnx_make_dummy_node('onnx/roialign', 'RoiAlign')
    for node in [gather1, squeeze, cast, gather, roialign, sub]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/gather1', 'onnx/squeeze')))
    edge_list.append(tuple(('onnx/squeeze', 'onnx/cast')))
    edge_list.append(tuple(('onnx/gather2', 'onnx/sub')))
    edge_list.append(tuple(('onnx/sub', 'onnx/roialign')))
    edge_list.append(tuple(('onnx/cast', 'onnx/roialign')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/gather1', 'onnx/roialign', 'onnx/gather2']

  def output_node(self):
    return "onnx/roialign"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing SF roi input layers")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "RoiAlign")
    _proto_copy_attribute(_data, node_def)
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    node_def_attr = helper.make_attribute("aligned", True)
    node_def.attribute.append(node_def_attr)
    nodes_to_remove.remove(node_def.name)
    return node_def


class CastReshape1_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    shape1 = onnx_make_dummy_node('onnx/shape1', 'Shape')
    gather1 = onnx_make_dummy_node('onnx/gather1', 'Gather')
    cast1 = onnx_make_dummy_node('onnx/cast1', 'Cast')
    slice1 = onnx_make_dummy_node('onnx/slice1', 'Slice')
    concat1 = onnx_make_dummy_node('onnx/concat1', 'Concat')
    cast2 = onnx_make_dummy_node('onnx/cast2', 'Cast')
    for node in [shape1, cast1, gather1, slice1, concat1, cast2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/shape1', 'onnx/gather1')))
    edge_list.append(tuple(('onnx/gather1', 'onnx/cast1')))
    edge_list.append(tuple(('onnx/cast1', 'onnx/slice1')))
    edge_list.append(tuple(('onnx/slice1', 'onnx/concat1')))
    edge_list.append(tuple(('onnx/concat1', 'onnx/cast2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/shape1']

  def output_node(self):
    return "onnx/cast2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("removing Shape->Gather->Cast->Slice->Concat->Cast")
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class CastReshape2_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    shape1 = onnx_make_dummy_node('onnx/shape1', 'Shape')
    cast1 = onnx_make_dummy_node('onnx/cast1', 'Cast')
    gather1 = onnx_make_dummy_node('onnx/gather1', 'Gather')
    concat1 = onnx_make_dummy_node('onnx/concat1', 'Concat')
    cast2 = onnx_make_dummy_node('onnx/cast2', 'Cast')
    for node in [shape1, cast1, gather1, concat1, cast2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/shape1', 'onnx/cast1')))
    edge_list.append(tuple(('onnx/cast1', 'onnx/gather1')))
    edge_list.append(tuple(('onnx/gather1', 'onnx/concat1')))
    edge_list.append(tuple(('onnx/concat1', 'onnx/cast2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/shape1']

  def output_node(self):
    return "onnx/cast2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("removing Shape->Cast->Gather->Concat->Cast")
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class ReshapeUnsqueeze_Matcher(BaseMatcher):

  def __init__(self, model, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    reshape1 = onnx_make_dummy_node('onnx/reshape1', 'Reshape')
    unsqueeze1 = onnx_make_dummy_node('onnx/unsqueeze1', 'Unsqueeze')
    for node in [reshape1, unsqueeze1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/reshape1', 'onnx/unsqueeze1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reshape1']

  def output_node(self):
    return "onnx/unsqueeze1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing Reshape->Unsqueeze")
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Reshape")
    self.shape_dict[matched_subgraph['onnx/reshape1'] + '_0'] = self.shape_dict[matched_subgraph['onnx/unsqueeze1'] + '_0']
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class GeluLayer_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    div = onnx_make_dummy_node('onnx/div_1', 'Div')
    erf = onnx_make_dummy_node('onnx/erf_1', 'Erf')
    add = onnx_make_dummy_node('onnx/add_1', 'Add')
    mul1 = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    mul2 = onnx_make_dummy_node('onnx/mul_2', 'Mul')
    for node in [div, erf, add, mul1, mul2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/div_1', 'onnx/erf_1')))
    edge_list.append(tuple(('onnx/erf_1', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/add_1', 'onnx/mul_1')))
    edge_list.append(tuple(('onnx/mul_1', 'onnx/mul_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/div_1', 'onnx/mul_1']

  def output_node(self):
    return "onnx/mul_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing nodes for GeLU")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    inputs_to_div1 = source_graph.nodes[matched_subgraph['onnx/div_1']]['data'].input
    inputs_to_mul1 = source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input
    if len(set(inputs_to_div1).intersection(set(inputs_to_mul1))) == 0:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Gelu")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class GeluLayer2_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    div = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    erf = onnx_make_dummy_node('onnx/erf_1', 'Erf')
    add = onnx_make_dummy_node('onnx/add_1', 'Add')
    mul1 = onnx_make_dummy_node('onnx/mul_2', 'Mul')
    mul2 = onnx_make_dummy_node('onnx/mul_3', 'Mul')
    for node in [div, erf, add, mul1, mul2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/mul_1', 'onnx/erf_1')))
    edge_list.append(tuple(('onnx/erf_1', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/add_1', 'onnx/mul_3')))
    edge_list.append(tuple(('onnx/mul_2', 'onnx/mul_3')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul_1', 'onnx/mul_2']

  def output_node(self):
    return "onnx/mul_3"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing nodes for GeLU")
    inputs_to_mul1 = source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input
    inputs_to_mul2 = source_graph.nodes[matched_subgraph['onnx/mul_2']]['data'].input
    if len(set(inputs_to_mul1).intersection(set(inputs_to_mul2))) == 0:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Gelu")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def

class GeluLayer3_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    div = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    erf = onnx_make_dummy_node('onnx/erf_1', 'Sigmoid')
    for node in [div, erf]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/mul_1', 'onnx/erf_1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul_1']

  def output_node(self):
    return "onnx/erf_1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing nodes for GeLU")
    inputs_to_mul1 = source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input
    if self.state_dict[inputs_to_mul1[1]] > 1.71 or self.state_dict[inputs_to_mul1[1]] < 1.7:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Gelu")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def    


class GeluLayer3_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    div = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    erf = onnx_make_dummy_node('onnx/erf_1', 'Sigmoid')
    for node in [div, erf]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/mul_1', 'onnx/erf_1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul_1']

  def output_node(self):
    return "onnx/erf_1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing nodes for GeLU")
    inputs_to_mul1 = source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input
    if self.state_dict[inputs_to_mul1[1]] > 1.71 or self.state_dict[inputs_to_mul1[1]] < 1.7:
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Gelu")
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class FullyConnected_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul1 = onnx_make_dummy_node('onnx/mul1', 'Mul')
    add1 = onnx_make_dummy_node('onnx/add1', 'Add')
    for node in [mul1, add1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/mul1', 'onnx/add1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul1']

  def output_node(self):
    return "onnx/add1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    inputs = source_graph.nodes[matched_subgraph['onnx/mul1']]['data'].input
    inputs_to_add = source_graph.nodes[matched_subgraph['onnx/add1']]['data'].input
    outputs = source_graph.nodes[matched_subgraph['onnx/add1']]['data'].output
    mul_output = source_graph.nodes[matched_subgraph['onnx/mul1']]['data'].output[0]
    # Replace the node only if one of the input to Mul layer is a blob
    # Input in self.state_dict => it is a blob
    if len(inputs) != 2 or inputs[1] not in self.state_dict:
      return None
    # Check if add layer has a constant input tensor
    if len(inputs_to_add) == 2 and inputs_to_add[1] not in self.state_dict:
      return None
    # Replace the node only if Add is the only output to Mul
    num_nodes = 0
    for node in source_graph.nodes:
      cur_node = source_graph.nodes[node]
      if 'data' not in cur_node.keys():
        continue
      all_fields = [field[0].name for field in cur_node['data'].ListFields()]
      if 'input' in all_fields and mul_output in cur_node['data'].input:
        num_nodes += 1
    if num_nodes != 1:
      return None
    input_shape = self.shape_dict[inputs[0]]
    output_shape = self.shape_dict[outputs[0]]
    if output_shape[2:] != [1] * (len(input_shape) - 2):
      return None
    # Return if Add node and Mul node is scalar
    if len(inputs_to_add) != 2 or inputs_to_add[1] in self.state_dict:
      if self.state_dict[inputs_to_add[1]].size == 1 and self.state_dict[inputs[1]].size == 1:
        return None
    bias_node = source_graph.nodes[matched_subgraph['onnx/add1']]['data'].input[1]
    new_bias_name = inputs[1].rstrip('_0') + '_1'
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "FullyConnected")
    node_def.input[:] = inputs_to_collapsed + [new_bias_name]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    node_def_attr = helper.make_attribute("flatten", False)
    node_def.attribute.append(node_def_attr)
    arr = self.state_dict[inputs[1]]
    zarr = np.zeros((arr.size, arr.size))
    np.fill_diagonal(zarr, arr)
    self.state_dict[inputs[1]] = zarr
    if bias_node in self.state_dict:
      self.state_dict[new_bias_name] = self.state_dict.pop(bias_node)
    return node_def


class Matmul_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    matmul1 = onnx_make_dummy_node('onnx/matmul1', 'MatMul')
    reshape1 = onnx_make_dummy_node('onnx/reshape1', 'Reshape')
    add1 = onnx_make_dummy_node('onnx/add1', 'Add')
    for node in [matmul1, reshape1, add1]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/matmul1', 'onnx/reshape1')))
    edge_list.append(tuple(('onnx/reshape1', 'onnx/add1')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/matmul1']

  def output_node(self):
    return "onnx/add1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    inputs = source_graph.nodes[matched_subgraph['onnx/matmul1']]['data'].input
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "FullyConnected")
    bias_node = source_graph.nodes[matched_subgraph['onnx/add1']]['data'].input[1]
    new_bias_name = inputs[1] + '_b'
    node_def.input[:] = inputs_to_collapsed + [new_bias_name]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    node_def_attr = helper.make_attribute("flatten", False)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("reshape", self.shape_dict[source_graph.nodes[matched_subgraph['onnx/reshape1']]['data'].output[0]])
    node_def.attribute.append(node_def_attr)
    if bias_node not in self.state_dict:
      return None
    #TODO: Zippin (4 lines)
    matmul_shape = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/matmul1']]['data'].output[0]]
    bias_shape = self.state_dict[bias_node].shape
    if len(bias_shape) != 2 or bias_shape[0] != 1 or bias_shape[1] != matmul_shape[-1]:
      return None
    self.state_dict[new_bias_name] = self.state_dict.pop(bias_node)
    return node_def


class LayerNormP1_Matcher(BaseMatcher):

  def __init__(self, model, state_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    reduce_mean1 = onnx_make_dummy_node('onnx/reduce_mean1', 'ReduceMean')
    sub1 = onnx_make_dummy_node('onnx/sub_1', 'Sub')
    sub2 = onnx_make_dummy_node('onnx/sub_2', 'Sub')
    pow1 = onnx_make_dummy_node('onnx/pow_1', 'Pow')
    reduce_mean2 = onnx_make_dummy_node('onnx/reduce_mean2', 'ReduceMean')
    add1 = onnx_make_dummy_node('onnx/add_1', 'Add')
    sqrt1 = onnx_make_dummy_node('onnx/sqrt_1', 'Sqrt')
    div1 = onnx_make_dummy_node('onnx/div_1', 'Div')
    mul1 = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    add2 = onnx_make_dummy_node('onnx/add_2', 'Add')
    for node in [reduce_mean1, sub1, sub2, pow1, reduce_mean2, add1, sqrt1, div1, mul1, add2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reduce_mean1', 'onnx/sub_1')))
    edge_list.append(tuple(('onnx/reduce_mean1', 'onnx/sub_2')))
    edge_list.append(tuple(('onnx/sub_2', 'onnx/pow_1')))
    edge_list.append(tuple(('onnx/pow_1', 'onnx/reduce_mean2')))
    edge_list.append(tuple(('onnx/reduce_mean2', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/add_1', 'onnx/sqrt_1')))
    edge_list.append(tuple(('onnx/sub_1', 'onnx/div_1')))
    edge_list.append(tuple(('onnx/sqrt_1', 'onnx/div_1')))
    edge_list.append(tuple(('onnx/div_1', 'onnx/mul_1')))
    edge_list.append(tuple(('onnx/mul_1', 'onnx/add_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reduce_mean1', 'onnx/sub_1', 'onnx/sub_2']

  def output_node(self):
    return "onnx/add_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing nodes for LayerNormP1")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    eps = self.state_dict[source_graph.nodes[matched_subgraph['onnx/add_1']]['data'].input[1]].tolist()
    beta = numpy_helper.from_array(self.state_dict[source_graph.nodes[matched_subgraph['onnx/add_2']]['data'].input[1]])
    gamma = numpy_helper.from_array(self.state_dict[source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input[0]])
    if len(source_graph.nodes[matched_subgraph['onnx/reduce_mean1']]['data'].attribute[0].ints) != 1 or len(source_graph.nodes[matched_subgraph['onnx/reduce_mean2']]['data'].attribute[0].ints) != 1:
      return None
    axes1 = source_graph.nodes[matched_subgraph['onnx/reduce_mean1']]['data'].attribute[0].ints[0]
    axes2 = source_graph.nodes[matched_subgraph['onnx/reduce_mean2']]['data'].attribute[0].ints[0]
    if axes1 != axes2:
      return None
    output_mean_var = False
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def_attr = helper.make_attribute("bias", beta)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("scale", gamma)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("eps", eps)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("axis", axes1)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("output_mean_var", output_mean_var)
    node_def.attribute.append(node_def_attr)
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class LayerNormP2_Matcher(BaseMatcher):

  def __init__(self, model, state_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    reduce_mean1 = onnx_make_dummy_node('onnx/reduce_mean1', 'ReduceMean')
    sub1 = onnx_make_dummy_node('onnx/sub_1', 'Sub')
    pow1 = onnx_make_dummy_node('onnx/pow_1', 'Pow')
    reduce_mean2 = onnx_make_dummy_node('onnx/reduce_mean2', 'ReduceMean')
    add1 = onnx_make_dummy_node('onnx/add_1', 'Add')
    sqrt1 = onnx_make_dummy_node('onnx/sqrt_1', 'Sqrt')
    div1 = onnx_make_dummy_node('onnx/div_1', 'Div')
    mul1 = onnx_make_dummy_node('onnx/mul_1', 'Mul')
    add2 = onnx_make_dummy_node('onnx/add_2', 'Add')
    for node in [reduce_mean1, sub1, pow1, reduce_mean2, add1, sqrt1, div1, mul1, add2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/reduce_mean1', 'onnx/sub_1')))
    edge_list.append(tuple(('onnx/sub_1', 'onnx/pow_1')))
    edge_list.append(tuple(('onnx/pow_1', 'onnx/reduce_mean2')))
    edge_list.append(tuple(('onnx/reduce_mean2', 'onnx/add_1')))
    edge_list.append(tuple(('onnx/add_1', 'onnx/sqrt_1')))
    edge_list.append(tuple(('onnx/sub_1', 'onnx/div_1')))
    edge_list.append(tuple(('onnx/sqrt_1', 'onnx/div_1')))
    edge_list.append(tuple(('onnx/div_1', 'onnx/mul_1')))
    edge_list.append(tuple(('onnx/mul_1', 'onnx/add_2')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/reduce_mean1']  #, 'onnx/sub_1']

  def output_node(self):
    return "onnx/add_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing nodes for LayerNormP2")
    if len(source_graph.nodes[matched_subgraph['onnx/reduce_mean1']]['data'].attribute[0].ints) != 1 or len(source_graph.nodes[matched_subgraph['onnx/reduce_mean2']]['data'].attribute[0].ints) != 1:
      return None
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    eps = self.state_dict[source_graph.nodes[matched_subgraph['onnx/add_1']]['data'].input[1]].tolist()
    beta = numpy_helper.from_array(self.state_dict[source_graph.nodes[matched_subgraph['onnx/add_2']]['data'].input[1]])
    gamma = numpy_helper.from_array(self.state_dict[source_graph.nodes[matched_subgraph['onnx/mul_1']]['data'].input[1]])
    axes1 = source_graph.nodes[matched_subgraph['onnx/reduce_mean1']]['data'].attribute[0].ints[0]
    axes2 = source_graph.nodes[matched_subgraph['onnx/reduce_mean2']]['data'].attribute[0].ints[0]
    if axes1 != axes2:
      return None
    output_mean_var = False
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def_attr = helper.make_attribute("bias", beta)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("scale", gamma)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("eps", eps)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("axis", axes1)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("output_mean_var", output_mean_var)
    node_def.attribute.append(node_def_attr)
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class L2NormMatcher_2(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    # node = onnx_make_dummy_node('onnx/Flatten', 'Flatten')
    # G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Pow_1', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ReduceSum', 'ReduceSum')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Pow_2', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    # edge_list.append(tuple(('onnx/Flatten', 'onnx/Pow_1')))
    edge_list.append(tuple(('onnx/Pow_1', 'onnx/ReduceSum')))
    edge_list.append(tuple(('onnx/ReduceSum', 'onnx/Pow_2')))
    edge_list.append(tuple(('onnx/Pow_2', 'onnx/Div')))
    # edge_list.append(tuple(('onnx/Flatten', 'onnx/Div')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Div']

  def output_node(self):
    return "onnx/Div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "L2Norm")
    node_def.attribute.append(helper.make_attribute("across_spatial", False))
    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def


class IBNMatcher(BaseMatcher):

  def __init__(self, model, state_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/Split', 'Split')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/InstanceNormalization', 'InstanceNormalization')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/BatchNormalization', 'BatchNormalization')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/Split', 'onnx/InstanceNormalization')))
    edge_list.append(tuple(('onnx/Split', 'onnx/BatchNormalization')))
    edge_list.append(tuple(('onnx/InstanceNormalization', 'onnx/Concat')))
    edge_list.append(tuple(('onnx/BatchNormalization', 'onnx/Concat')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Split']

  def output_node(self):
    return "onnx/Concat"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    inorm_node = source_graph.nodes[matched_subgraph['onnx/InstanceNormalization']]['data']
    bnorm_node = source_graph.nodes[matched_subgraph['onnx/BatchNormalization']]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "IBN")
    in_eps = next(i for i in source_graph.nodes[matched_subgraph['onnx/InstanceNormalization']]['data'].attribute if i.name == 'epsilon').f
    bn_eps = next(i for i in source_graph.nodes[matched_subgraph['onnx/BatchNormalization']]['data'].attribute if i.name == 'epsilon').f
    node_def.attribute.append(helper.make_attribute("in_eps", in_eps))
    node_def.attribute.append(helper.make_attribute("bn_eps", bn_eps))
    node_def.input[:] = source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data'].input[:1]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    node_def.input.extend(inorm_node.input[1:])
    mean = self.state_dict[bnorm_node.input[3]]
    var = self.state_dict[bnorm_node.input[4]]
    scale = self.state_dict[bnorm_node.input[1]]
    bias = self.state_dict[bnorm_node.input[2]]
    self.state_dict[inorm_node.input[1]] = np.concatenate((self.state_dict[inorm_node.input[1]], scale / np.sqrt(var + bn_eps)), axis=0)
    self.state_dict[inorm_node.input[2]] = np.concatenate((self.state_dict[inorm_node.input[2]], bias - ((mean * scale) / np.sqrt(var + bn_eps))), axis=0)
    return node_def


class ConcatUpsample_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    concat = onnx_make_dummy_node('onnx/Concat', 'Concat')
    upsample = onnx_make_dummy_node('onnx/Upsample', 'Upsample')
    for node in [concat, upsample]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/Concat', 'onnx/Upsample')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Upsample']

  def output_node(self):
    return "onnx/Upsample"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # ADI
    # TODO: condition: all input to concat is "Constant"
    # ..
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Upsample", _out=_data.output)
    # mul_initializer = source_graph.nodes[matched_subgraph['onnx/Upsample']]['data'].input[1]
    _proto_copy_attribute(source_graph.nodes[matched_subgraph['onnx/Upsample']]['data'], node_def)
    inputs_to_collapsed.remove(source_graph.nodes[matched_subgraph['onnx/Concat']]['data'].name)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class AliaswithNameRoi_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    alias = onnx_make_dummy_node('onnx/aliaswithname', 'AliasWithName')
    roi = onnx_make_dummy_node('onnx/roialign', 'RoIAlign')
    for node in [roi, alias]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/aliaswithname', 'onnx/roialign')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/roialign', 'onnx/aliaswithname']

  def output_node(self):
    return "onnx/roialign"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "RoIAlign", _out=_data.output)
    _proto_copy_attribute(source_graph.nodes[matched_subgraph['onnx/roialign']]['data'], node_def)
    inputs_to_collapsed = set(inputs_to_collapsed) - set(nodes_to_remove)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class DensePose_Upsample_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    shape = onnx_make_dummy_node('onnx/shape', 'Shape')
    shape1 = onnx_make_dummy_node('onnx/shape_1', 'Shape')
    gather = onnx_make_dummy_node('onnx/gather', 'Gather')
    gather1 = onnx_make_dummy_node('onnx/gather_1', 'Gather')
    us = onnx_make_dummy_node('onnx/us', 'Unsqueeze')
    us1 = onnx_make_dummy_node('onnx/us_1', 'Unsqueeze')
    concat = onnx_make_dummy_node("onnx/concat", "Concat")
    cast = onnx_make_dummy_node("onnx/cast", "Cast")
    cast1 = onnx_make_dummy_node("onnx/cast_1", "Cast")
    shape2 = onnx_make_dummy_node("onnx/shape_2", "Shape")
    slic = onnx_make_dummy_node("onnx/slice", "Slice")
    div = onnx_make_dummy_node("onnx/div", "Div")
    concat1 = onnx_make_dummy_node("onnx/concat_1", "Concat")
    upsample = onnx_make_dummy_node('onnx/upsample', 'Upsample')
    for node in [shape, shape1, gather, gather1, us, us1, concat, cast, cast1, shape2, slic, div, concat1, upsample]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/shape', 'onnx/gather')))
    edge_list.append(tuple(('onnx/shape_1', 'onnx/gather_1')))
    edge_list.append(tuple(('onnx/gather', 'onnx/us')))
    edge_list.append(tuple(('onnx/gather_1', 'onnx/us_1')))
    edge_list.append(tuple(('onnx/us', 'onnx/concat')))
    edge_list.append(tuple(('onnx/us_1', 'onnx/concat')))
    edge_list.append(tuple(('onnx/concat', 'onnx/cast')))
    edge_list.append(tuple(('onnx/shape_2', 'onnx/slice')))
    edge_list.append(tuple(('onnx/slice', 'onnx/cast_1')))
    edge_list.append(tuple(('onnx/cast', 'onnx/div')))
    edge_list.append(tuple(('onnx/cast_1', 'onnx/div')))
    edge_list.append(tuple(('onnx/div', 'onnx/concat_1')))
    edge_list.append(tuple(('onnx/concat_1', 'onnx/upsample')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/upsample']

  def output_node(self):
    return "onnx/upsample"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Upsample", _out=_data.output)
    _proto_copy_attribute(source_graph.nodes[matched_subgraph['onnx/upsample']]['data'], node_def)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class GroupnormCast_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    gn = onnx_make_dummy_node("onnx/aten", "ATen")
    cast1 = onnx_make_dummy_node("onnx/cast1", "Cast")  #endswith weight
    cast2 = onnx_make_dummy_node("onnx/cast2", "Cast")  #endswith bias
    for node in [gn, cast1, cast2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(("onnx/cast1", "onnx/aten")))
    edge_list.append(tuple(("onnx/cast2", "onnx/aten")))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/aten']

  def output_node(self):
    return 'onnx/aten'

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "ATen", _out=_data.output)
    _proto_copy_attribute(source_graph.nodes[matched_subgraph['onnx/aten']]['data'], node_def)
    _input_weigths = _data.input[1] if len(_data.input) > 1 else None
    _input_bias = _data.input[2] if len(_data.input) > 2 else None
    self.state_dict[_input_weigths] = self.state_dict[source_graph.nodes[_input_weigths]["data"].input[0]]
    self.state_dict[_input_bias] = self.state_dict[source_graph.nodes[_input_bias]["data"].input[0]]
    node_def.input[:] = inputs_to_collapsed
    return node_def


class WeedOut(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    us1 = onnx_make_dummy_node("onnx/us1", "Unsqueeze")
    us2 = onnx_make_dummy_node("onnx/us2", "Unsqueeze")
    us3 = onnx_make_dummy_node("onnx/us3", "Unsqueeze")
    us4 = onnx_make_dummy_node("onnx/us4", "Unsqueeze")
    us5 = onnx_make_dummy_node("onnx/us5", "Unsqueeze")
    us6 = onnx_make_dummy_node("onnx/us6", "Unsqueeze")
    us7 = onnx_make_dummy_node("onnx/us7", "Unsqueeze")
    us8 = onnx_make_dummy_node("onnx/us8", "Unsqueeze")
    shape1 = onnx_make_dummy_node("onnx/shape1", "Shape")
    shape2 = onnx_make_dummy_node("onnx/shape2", "Shape")
    ct = onnx_make_dummy_node("onnx/ct", "ConvTranspose")
    us = onnx_make_dummy_node("onnx/us", "Upsample")
    sp = onnx_make_dummy_node("onnx/sp", "Softplus")
    g = onnx_make_dummy_node("onnx/g", "Gather")
    c1 = onnx_make_dummy_node("onnx/c1", "Cast")
    c2 = onnx_make_dummy_node("onnx/c2", "Cast")
    a = onnx_make_dummy_node("onnx/add", "Add")
    e = onnx_make_dummy_node("onnx/expand", "Expand")
    con3 = onnx_make_dummy_node("onnx/con3", "Concat")
    t = onnx_make_dummy_node("onnx/t", "Tile")
    con1 = onnx_make_dummy_node("onnx/con1", "Concat")
    con2 = onnx_make_dummy_node("onnx/con2", "Concat")
    m = onnx_make_dummy_node("onnx/m", "Mul")
    cos = onnx_make_dummy_node("onnx/cos", "ConstantOfShape")
    for node in [us1, us2, us3, us4, us5, us6, us7, us8, shape1, shape2, ct, us, sp, g, c1, c2, a, e, t, m, con1, con2, cos, con3]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(("onnx/shape1", "onnx/g")))
    edge_list.append(tuple(("onnx/g", "onnx/c1")))
    edge_list.append(tuple(("onnx/c1", "onnx/c2")))
    edge_list.append(tuple(("onnx/c2", "onnx/us2")))
    edge_list.append(tuple(("onnx/c2", "onnx/us6")))
    edge_list.append(tuple(("onnx/us1", "onnx/con1")))
    edge_list.append(tuple(("onnx/us3", "onnx/con1")))
    edge_list.append(tuple(("onnx/us4", "onnx/con1")))
    edge_list.append(tuple(("onnx/us5", "onnx/con2")))
    edge_list.append(tuple(("onnx/us7", "onnx/con2")))
    edge_list.append(tuple(("onnx/us8", "onnx/con2")))
    edge_list.append(tuple(("onnx/us2", "onnx/con1")))
    edge_list.append(tuple(("onnx/us6", "onnx/con2")))
    edge_list.append(tuple(("onnx/ct", "onnx/us")))
    edge_list.append(tuple(("onnx/us", "onnx/sp")))
    edge_list.append(tuple(("onnx/sp", "onnx/add")))
    edge_list.append(tuple(("onnx/add", "onnx/expand")))
    edge_list.append(tuple(("onnx/con2", "onnx/shape2")))
    edge_list.append(tuple(("onnx/cos", "onnx/expand")))
    edge_list.append(tuple(("onnx/shape2", "onnx/cos")))
    edge_list.append(tuple(("onnx/con3", "onnx/us")))
    edge_list.append(tuple(("onnx/con1", "onnx/t")))
    edge_list.append(tuple(("onnx/expand", "onnx/t")))
    edge_list.append(tuple(("onnx/t", "onnx/m")))
    G.add_edges_from(edge_list)
    write_dot(G, "t.dot")
    return G

  def input_nodes(self):
    return ['onnx/shape1']

  def output_node(self):
    return 'onnx/m'

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Identity", _out=_data.output)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class GCD_Matcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    gp = onnx_make_dummy_node('onnx/gp4', 'GenerateProposals')
    gp0 = onnx_make_dummy_node('onnx/gp0', 'GenerateProposals')
    gp1 = onnx_make_dummy_node('onnx/gp1', 'GenerateProposals')
    gp2 = onnx_make_dummy_node('onnx/gp2', 'GenerateProposals')
    gp3 = onnx_make_dummy_node('onnx/gp3', 'GenerateProposals')
    cp = onnx_make_dummy_node('onnx/cp', 'CollectRpnProposals')
    dp = onnx_make_dummy_node('onnx/dp', 'DistributeFpnProposals')
    for node in [gp, gp0, gp1, gp2, gp3, dp, cp]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/gp0', 'onnx/cp')))
    edge_list.append(tuple(('onnx/gp1', 'onnx/cp')))
    edge_list.append(tuple(('onnx/gp2', 'onnx/cp')))
    edge_list.append(tuple(('onnx/gp3', 'onnx/cp')))
    edge_list.append(tuple(('onnx/gp4', 'onnx/cp')))
    edge_list.append(tuple(('onnx/cp', 'onnx/dp')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/gp0', 'onnx/gp1', 'onnx/gp2', 'onnx/gp3', 'onnx/gp4']

  def output_node(self):
    return 'onnx/dp'

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # ADI
    # TODO: condition: all input to concat is "Constant"
    # ..
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "GCD", _out=_data.output)
    # mul_initializer = source_graph.nodes[matched_subgraph['onnx/Upsample']]['data'].input[1]
    _proto_copy_attribute(source_graph.nodes[matched_subgraph['onnx/gp0']]['data'], node_def)
    ss = []
    for i in range(5):
      _node_name = "onnx/gp{}".format(i)
      data = source_graph.nodes[matched_subgraph[_node_name]]['data']
      ss.append(self.get_float_attribute(data, "spatial_scale"))
    value = helper.make_attribute('spatial_scales', ss)
    node_def.attribute.MergeFrom([value])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class RepeatInterleave_Matcher(BaseMatcher):

  def __init__(self, model):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/Shape_1', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast_1', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast_2', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_1', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_2', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_3', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_4', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_1', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_5', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_6', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_7', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_8', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_2', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape_2', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ConstantOfShape', 'ConstantOfShape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Expand', 'Expand')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Tile', 'Tile')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/Shape_1', 'onnx/Gather')))
    edge_list.append(tuple(('onnx/Gather', 'onnx/Cast_1')))
    edge_list.append(tuple(('onnx/Cast_1', 'onnx/Cast_2')))
    edge_list.append(tuple(('onnx/Cast_2', 'onnx/Unsqueeze_1')))
    edge_list.append(tuple(('onnx/Unsqueeze_1', 'onnx/Concat_1')))
    edge_list.append(tuple(('onnx/Unsqueeze_2', 'onnx/Concat_1')))
    edge_list.append(tuple(('onnx/Unsqueeze_3', 'onnx/Concat_1')))
    edge_list.append(tuple(('onnx/Unsqueeze_4', 'onnx/Concat_1')))
    edge_list.append(tuple(('onnx/Cast_2', 'onnx/Unsqueeze_5')))
    edge_list.append(tuple(('onnx/Unsqueeze_5', 'onnx/Concat_2')))
    edge_list.append(tuple(('onnx/Unsqueeze_6', 'onnx/Concat_2')))
    edge_list.append(tuple(('onnx/Unsqueeze_7', 'onnx/Concat_2')))
    edge_list.append(tuple(('onnx/Unsqueeze_8', 'onnx/Concat_2')))
    edge_list.append(tuple(('onnx/Concat_2', 'onnx/Shape_2')))
    edge_list.append(tuple(('onnx/Shape_2', 'onnx/ConstantOfShape')))
    edge_list.append(tuple(('onnx/ConstantOfShape', 'onnx/Expand')))
    edge_list.append(tuple(('onnx/Expand', 'onnx/Tile')))
    edge_list.append(tuple(('onnx/Concat_1', 'onnx/Tile')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape_1']

  def output_node(self):
    return "onnx/Tile"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    expand_node = source_graph.nodes[matched_subgraph['onnx/Expand']]['data']
    output_node = source_graph.nodes[matched_subgraph['onnx/Tile']]['data']
    out_mul_node = source_graph.nodes[[i for i in source_graph.successors(output_node.name)][0]]['data']
    for i, j in enumerate(out_mul_node.input):
      if j == output_node.output[0]:
        out_mul_node.input[i] = expand_node.input[0]
    for i in matched_subgraph.values():
      source_graph.remove_node(i)
    return None


class GroupNorm_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/Reshape_1', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/InstanceNormalization', 'InstanceNormalization')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Reshape_2', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/Reshape_1', 'onnx/InstanceNormalization')))
    edge_list.append(tuple(('onnx/InstanceNormalization', 'onnx/Reshape_2')))
    edge_list.append(tuple(('onnx/Shape', 'onnx/Reshape_2')))
    edge_list.append(tuple(('onnx/Reshape_2', 'onnx/Mul')))
    edge_list.append(tuple(('onnx/Mul', 'onnx/Add')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape', 'onnx/Reshape_1']

  def output_node(self):
    return "onnx/Add"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    INorm_wgts = source_graph.nodes[matched_subgraph['onnx/InstanceNormalization']]['data'].input[1:]
    if any([i != 1.0 for i in self.state_dict[INorm_wgts[0]]]) or any([i != 0.0 for i in self.state_dict[INorm_wgts[1]]]):
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "GroupNorm")
    node_def.input[:] = source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data'].input
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    num_groups = self.state_dict[source_graph.nodes[matched_subgraph['onnx/Reshape_1']]['data'].input[1]][1]
    node_def.input.append(source_graph.nodes[matched_subgraph['onnx/Mul']]['data'].input[1])
    node_def.input.append(source_graph.nodes[matched_subgraph['onnx/Add']]['data'].input[1])
    node_def.attribute.append(helper.make_attribute("groups", num_groups))
    eps = next(i for i in source_graph.nodes[matched_subgraph['onnx/InstanceNormalization']]['data'].attribute if i.name == 'epsilon').f
    node_def.attribute.append(helper.make_attribute("eps", eps))
    return node_def


class GroupNorm_1_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Reshape_1', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/InstanceNormalization', 'InstanceNormalization')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Reshape_2', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Reshape_1', 'onnx/InstanceNormalization')))
    edge_list.append(tuple(('onnx/InstanceNormalization', 'onnx/Reshape_2')))
    edge_list.append(tuple(('onnx/Reshape_2', 'onnx/Mul')))
    edge_list.append(tuple(('onnx/Mul', 'onnx/Add')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Reshape_1']

  def output_node(self):
    return "onnx/Add"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    INorm_wgts = source_graph.nodes[matched_subgraph['onnx/InstanceNormalization']]['data'].input[1:]
    if any([i != 1.0 for i in self.state_dict[INorm_wgts[0]]]) or any([i != 0.0 for i in self.state_dict[INorm_wgts[1]]]):
      return None
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "GroupNorm")
    node_def.input[:] = source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data'].input[:1]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    num_groups = self.state_dict[source_graph.nodes[matched_subgraph['onnx/Reshape_1']]['data'].input[1]][1]
    node_def.input.append(source_graph.nodes[matched_subgraph['onnx/Mul']]['data'].input[1])
    node_def.input.append(source_graph.nodes[matched_subgraph['onnx/Add']]['data'].input[1])
    node_def.attribute.append(helper.make_attribute("groups", num_groups))
    eps = next(i for i in source_graph.nodes[matched_subgraph['onnx/InstanceNormalization']]['data'].attribute if i.name == 'epsilon').f
    node_def.attribute.append(helper.make_attribute("eps", eps))
    return node_def


class Tile_Mul_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/Tile', 'Tile')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/Tile', 'onnx/Mul')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Tile']

  def output_node(self):
    return "onnx/Mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    mul_node = source_graph.nodes[matched_subgraph['onnx/Mul']]['data']
    tile_node = source_graph.nodes[matched_subgraph['onnx/Tile']]['data']
    if mul_node.input[0] not in self.shape_dict or mul_node.input[1] not in self.shape_dict:
      return None
    if tile_node.input[0] not in self.shape_dict:
      return None
    for i, j in enumerate(mul_node.input):
      if j == tile_node.output[0]:
        mul_node.input[i] = tile_node.input[0]
    source_graph.add_edges_from([(tile_node.input[0], matched_subgraph['onnx/Mul'])])
    source_graph.remove_node(matched_subgraph['onnx/Tile'])
    return None


class Flynn_d_tail_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = onnx_make_dummy_node('onnx/Shape_1', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_1', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_1', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_2', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_2', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape_2', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ConstantOfShape', 'ConstantOfShape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Expand', 'Expand')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Tile', 'Tile')
    G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx/Shape_1', 'onnx/Gather')))
    edge_list.append(tuple(('onnx/Gather', 'onnx/Cast')))
    edge_list.append(tuple(('onnx/Cast', 'onnx/Unsqueeze_1')))
    edge_list.append(tuple(('onnx/Unsqueeze_1', 'onnx/Concat_1')))
    edge_list.append(tuple(('onnx/Cast', 'onnx/Unsqueeze_2')))
    edge_list.append(tuple(('onnx/Unsqueeze_2', 'onnx/Concat_2')))
    edge_list.append(tuple(('onnx/Concat_2', 'onnx/Shape_2')))
    edge_list.append(tuple(('onnx/Shape_2', 'onnx/ConstantOfShape')))
    edge_list.append(tuple(('onnx/ConstantOfShape', 'onnx/Expand')))
    edge_list.append(tuple(('onnx/Concat_1', 'onnx/Tile')))
    edge_list.append(tuple(('onnx/Expand', 'onnx/Tile')))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape_1']

  def output_node(self):
    return "onnx/Tile"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    expand_node = source_graph.nodes[matched_subgraph['onnx/Expand']]['data']
    COS_node = source_graph.nodes[matched_subgraph['onnx/ConstantOfShape']]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Identity")
    COS_output = [i for i in model.graph.node if i.name == COS_node.name][0].output
    node_inp = [i for i in expand_node.input if i not in COS_output]
    node_def.input[:] = node_inp
    node_def.output[:] = source_graph.nodes[matched_subgraph['onnx/Tile']]['data'].output
    return node_def


class Mul2_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Mul_1', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_2', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Mul_1', 'onnx/Mul_2')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Mul_1']

  def output_node(self):
    return "onnx/Mul_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    mul1_node = source_graph.nodes[matched_subgraph['onnx/Mul_1']]['data']
    mul2_node = source_graph.nodes[matched_subgraph['onnx/Mul_2']]['data']
    if (mul1_node.input[1] not in self.state_dict) or (mul2_node.input[1] not in self.state_dict):
      return None
    mul1_const = self.state_dict[mul1_node.input[1]][()]
    mul2_const = self.state_dict[mul2_node.input[1]][()]
    new_const = mul1_const * mul2_const
    new_const_name = mul1_node.input[1] + '_const_1'
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Mul")
    node_def.input[:] = inputs_to_collapsed[:-1] + [new_const_name]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    self.state_dict[new_const_name] = new_const
    return node_def


def onnx_subGraph_match_replace_L2NormScale_Matcher(G, model, shape_dict, state_dict):
  return L2NormScale_Matcher(model, shape_dict, state_dict).replace_subgraph_instances(G, model)


class NullPattern_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Mul_1', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Sqrt', 'Sqrt')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_1', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Log', 'Log')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_2', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Floor', 'Floor')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Max', 'Max')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Min', 'Min')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Mul_1', 'onnx/Sqrt')))
    edge_list.append(tuple(('onnx/Sqrt', 'onnx/Div_1')))
    edge_list.append(tuple(('onnx/Div_1', 'onnx/Log')))
    edge_list.append(tuple(('onnx/Log', 'onnx/Div_2')))
    edge_list.append(tuple(('onnx/Div_2', 'onnx/Floor')))
    edge_list.append(tuple(('onnx/Floor', 'onnx/Add')))
    edge_list.append(tuple(('onnx/Add', 'onnx/Cast')))
    edge_list.append(tuple(('onnx/Cast', 'onnx/Max')))
    edge_list.append(tuple(('onnx/Max', 'onnx/Min')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Mul_1']

  def output_node(self):
    return "onnx/Min"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    max_node = source_graph.nodes[matched_subgraph['onnx/Max']]['data']
    min_node = source_graph.nodes[matched_subgraph['onnx/Min']]['data']
    if (max_node.input[1] not in self.state_dict) or (min_node.input[1] not in self.state_dict):
      return None
    min_const = self.state_dict[min_node.input[1]].tolist()
    max_const = self.state_dict[max_node.input[1]].tolist()
    if min_const != 0 and max_const != 0:
      return None
    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.attribute.append(helper.make_attribute("value", 0))
    node_def.input[:] = []
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    self.state_dict[output_node_name] = np.zeros(self.shape_dict[output_node_name], dtype=np.int32)
    return node_def


class ConstCast_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Cast', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Cast']

  def output_node(self):
    return "onnx/Cast"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    input_node_name = source_graph.nodes[matched_subgraph['onnx/Cast']]['data'].input[0]
    if input_node_name not in self.state_dict:
      return None
    input_node_val = self.state_dict[input_node_name]
    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.attribute.append(helper.make_attribute("value", 0))
    node_def.input[:] = []
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    self.state_dict[output_node_name] = input_node_val
    return node_def


class ConstPow_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Pow', 'Pow')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Pow']

  def output_node(self):
    return "onnx/Pow"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):

    def get_constant(val):
      if isinstance(val, float) or isinstance(val, int):
        return val
      elif isinstance(val, list):
        return get_constant(val[0])

    input_node_name = source_graph.nodes[matched_subgraph['onnx/Pow']]['data'].input[1]
    if input_node_name not in self.state_dict:
      return None
    y = self.state_dict[input_node_name]
    if source_graph.nodes[matched_subgraph['onnx/Pow']]['data'].input[0] not in self.state_dict:
      return None
    x = self.state_dict[source_graph.nodes[matched_subgraph['onnx/Pow']]['data'].input[0]].tolist()
    x = get_constant(x)
    new_val = np.power(x, y)
    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.attribute.append(helper.make_attribute("value", 0))
    node_def.input[:] = []
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    self.state_dict[output_node_name] = np.array(new_val)
    return node_def


class ConstGather_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Gather', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Gather']

  def output_node(self):
    return "onnx/Gather"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    data = source_graph.nodes[matched_subgraph['onnx/Gather']]['data'].input[0]
    indices = source_graph.nodes[matched_subgraph['onnx/Gather']]['data'].input[1]
    if data not in self.state_dict or indices not in self.state_dict:
      return None
    data = self.state_dict[data]
    indices = self.state_dict[indices]
    axis = [attr.i for attr in source_graph.nodes[matched_subgraph['onnx/Gather']]['data'].attribute if attr.name == "axis"]
    axis = axis[0] if len(axis) == 1 else None
    output = np.take(data, indices, axis)
    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.attribute.append(helper.make_attribute("value", 0))
    node_def.input[:] = []
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    self.state_dict[output_node_name] = output
    return node_def


class ConstReshape_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Reshape', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Reshape']

  def output_node(self):
    return "onnx/Reshape"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    input_node = source_graph.nodes[matched_subgraph['onnx/Reshape']]['data'].input[0]
    if input_node not in self.state_dict:
      return None
    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.attribute.append(helper.make_attribute("value", 0))
    node_def.input[:] = []
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    self.state_dict[output_node_name] = np.array(self.state_dict[input_node]).reshape(self.shape_dict[output_node_name])
    return node_def


class ConstUnsqueeze_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Unsqueeze', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Unsqueeze']

  def output_node(self):
    return "onnx/Unsqueeze"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    input_node_name = source_graph.nodes[matched_subgraph['onnx/Unsqueeze']]['data'].input[0]
    if input_node_name not in self.state_dict:
      return None
    const_val = self.state_dict[input_node_name]

    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
    node_def.attribute.append(helper.make_attribute("value", 0))
    node_def.input[:] = []
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

    self.state_dict[output_node_name] = np.array(const_val).reshape(self.shape_dict[output_node_name])
    return node_def


class ConstDiv_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Div', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Div']

  def output_node(self):
    return "onnx/Div"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):

    def get_constant(val):
      if isinstance(val, float) or isinstance(val, int):
        return val
      elif isinstance(val, list):
        return get_constant(val[0])

    input_node_name = source_graph.nodes[matched_subgraph['onnx/Div']]['data'].input[1]
    if input_node_name not in self.state_dict:
      return None
    y = self.state_dict[input_node_name]
    if not np.all(y == 1):
      return None
    input1_node_name = source_graph.nodes[matched_subgraph['onnx/Div']]['data'].input[0]
    output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
    node_def = None
    if input1_node_name in self.state_dict:
      x = self.state_dict[input1_node_name].tolist()
      x = get_constant(x)
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "DVTensor")
      node_def.attribute.append(helper.make_attribute("value", x))
      node_def.input[:] = []
      node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    else:
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
      node_def.attribute.append(helper.make_attribute("value", 0))
      node_def.input[:] = []
      node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
      for node in source_graph.nodes:
        cur_node = source_graph.nodes[node]
        if 'input' in cur_node['data'].__str__() and output_node_name in cur_node['data'].input:
          cur_node['data'].input[:] = [input1_node_name if op == output_node_name else op for op in cur_node['data'].input]
          break

    return node_def


class ConstAdd_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Add', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Add']

  def output_node(self):
    return "onnx/Add"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    input0_node_name = source_graph.nodes[matched_subgraph['onnx/Add']]['data'].input[0]
    input1_node_name = source_graph.nodes[matched_subgraph['onnx/Add']]['data'].input[1]
    if input0_node_name in self.state_dict and input1_node_name in self.state_dict:
      input0 = self.state_dict[input0_node_name]
      input1 = self.state_dict[input1_node_name]
      output = input0 + input1

      output_node_name = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
      node_def.attribute.append(helper.make_attribute("value", 0))
      node_def.input[:] = []
      node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output

      self.state_dict[output_node_name] = np.array(output).reshape(self.shape_dict[output_node_name])
      return node_def
    return None


class ZeroTensorAdd_Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Add', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Add']

  def output_node(self):
    return "onnx/Add"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    input0_node_name = source_graph.nodes[matched_subgraph['onnx/Add']]['data'].input[0]
    input1_node_name = source_graph.nodes[matched_subgraph['onnx/Add']]['data'].input[1]
    const_input = None
    other_input = None
    if input0_node_name in self.state_dict:
      const_input = self.state_dict[input0_node_name]
      other_input = input1_node_name
    elif input1_node_name in self.state_dict:
      const_input = self.state_dict[input1_node_name]
      other_input = input0_node_name
    else:
      return None
    if np.all(const_input == 0):
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "dv_remove")
      node_def.attribute.append(helper.make_attribute("value", 0))
      node_def.input[:] = [other_input]
      node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
      return node_def
    return None


class Unet1Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Shape_1', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_2', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_3', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast_4', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast_5', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_6', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_7', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast_8', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Cast_9', 'Cast')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_10', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)

    node = onnx_make_dummy_node('onnx/Shape_11', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_12', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_13', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_14', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_15', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_16', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Reshape_17', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Transpose_18', 'Transpose')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Reshape_19', 'Reshape')
    G.add_node(node.name, data=node, type=node.op_type)

    node = onnx_make_dummy_node('onnx/Shape_20', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_21', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_22', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_23', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_24', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Shape_1', 'onnx/Gather_2')))
    edge_list.append(tuple(('onnx/Gather_2', 'onnx/Div_3')))
    edge_list.append(tuple(('onnx/Div_3', 'onnx/Cast_4')))
    edge_list.append(tuple(('onnx/Cast_4', 'onnx/Cast_5')))
    edge_list.append(tuple(('onnx/Cast_5', 'onnx/Unsqueeze_6')))
    edge_list.append(tuple(('onnx/Unsqueeze_6', 'onnx/Concat_14')))
    edge_list.append(tuple(('onnx/Gather_2', 'onnx/Div_7')))
    edge_list.append(tuple(('onnx/Div_7', 'onnx/Cast_8')))
    edge_list.append(tuple(('onnx/Cast_8', 'onnx/Cast_9')))
    edge_list.append(tuple(('onnx/Cast_9', 'onnx/Unsqueeze_10')))
    edge_list.append(tuple(('onnx/Unsqueeze_10', 'onnx/Concat_16')))

    edge_list.append(tuple(('onnx/Shape_11', 'onnx/Gather_12')))
    edge_list.append(tuple(('onnx/Gather_12', 'onnx/Unsqueeze_13')))
    edge_list.append(tuple(('onnx/Unsqueeze_13', 'onnx/Concat_14')))
    edge_list.append(tuple(('onnx/Concat_14', 'onnx/Reshape_19')))
    edge_list.append(tuple(('onnx/Gather_12', 'onnx/Unsqueeze_15')))
    edge_list.append(tuple(('onnx/Unsqueeze_15', 'onnx/Concat_16')))
    edge_list.append(tuple(('onnx/Concat_16', 'onnx/Reshape_17')))
    edge_list.append(tuple(('onnx/Reshape_17', 'onnx/Transpose_18')))
    edge_list.append(tuple(('onnx/Transpose_18', 'onnx/Reshape_19')))

    edge_list.append(tuple(('onnx/Shape_20', 'onnx/Gather_21')))
    edge_list.append(tuple(('onnx/Gather_21', 'onnx/Mul_22')))
    edge_list.append(tuple(('onnx/Mul_22', 'onnx/Unsqueeze_23')))
    edge_list.append(tuple(('onnx/Unsqueeze_23', 'onnx/Concat_14')))
    edge_list.append(tuple(('onnx/Gather_21', 'onnx/Unsqueeze_24')))
    edge_list.append(tuple(('onnx/Unsqueeze_24', 'onnx/Concat_16')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape_1']

  def output_node(self):
    return "onnx/Reshape_19"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    reshape_1_node = source_graph.nodes[matched_subgraph['onnx/Reshape_17']]['data']
    transpose_node = source_graph.nodes[matched_subgraph['onnx/Transpose_18']]['data']
    reshape_2_node = source_graph.nodes[matched_subgraph['onnx/Reshape_19']]['data']

    reshape_1_shape = self.shape_dict[reshape_1_node.output[0]]
    reshape_2_shape = self.shape_dict[reshape_2_node.output[0]]

    final_nodes = [reshape_1_node.name, transpose_node.name, reshape_2_node.name]
    all_nodes = [i for i in matched_subgraph.values() if i not in final_nodes]

    for i in all_nodes:
      source_graph.remove_node(i)
    source_graph.add_edges_from([(inputs_to_collapsed[0], reshape_1_node.name)])
    self.state_dict[reshape_1_node.input[1]] = np.array(reshape_1_shape)
    self.state_dict[reshape_2_node.input[1]] = np.array(reshape_2_shape)
    logger.info("Unet pattern 1 removed nodes: " + ','.join(i for i in matched_subgraph.values()))
    return None


class Unet2Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Shape_1', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_2', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_3', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape_4', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_5', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_6', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape_7', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_8', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_9', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_10', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/ConstantOfShape_11', 'ConstantOfShape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_12', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Shape_1', 'onnx/Gather_2')))
    edge_list.append(tuple(('onnx/Gather_2', 'onnx/Unsqueeze_3')))
    edge_list.append(tuple(('onnx/Shape_4', 'onnx/Gather_5')))
    edge_list.append(tuple(('onnx/Gather_5', 'onnx/Unsqueeze_6')))
    edge_list.append(tuple(('onnx/Shape_7', 'onnx/Gather_8')))
    edge_list.append(tuple(('onnx/Gather_8', 'onnx/Unsqueeze_9')))

    edge_list.append(tuple(('onnx/Unsqueeze_3', 'onnx/Concat_10')))
    edge_list.append(tuple(('onnx/Unsqueeze_6', 'onnx/Concat_10')))
    edge_list.append(tuple(('onnx/Unsqueeze_9', 'onnx/Concat_10')))
    edge_list.append(tuple(('onnx/Concat_10', 'onnx/ConstantOfShape_11')))
    edge_list.append(tuple(('onnx/ConstantOfShape_11', 'onnx/Mul_12')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape_1', 'onnx/Shape_4', 'onnx/Shape_7']

  def output_node(self):
    return "onnx/Mul_12"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    shape_1_node = source_graph.nodes[matched_subgraph['onnx/Shape_1']]['data']
    shape_2_node = source_graph.nodes[matched_subgraph['onnx/Shape_4']]['data']
    shape_3_node = source_graph.nodes[matched_subgraph['onnx/Shape_7']]['data']
    unsqueeze_1_node = source_graph.nodes[matched_subgraph['onnx/Unsqueeze_3']]['data']
    unsqueeze_2_node = source_graph.nodes[matched_subgraph['onnx/Unsqueeze_6']]['data']
    unsqueeze_3_node = source_graph.nodes[matched_subgraph['onnx/Unsqueeze_9']]['data']
    concat_node = source_graph.nodes[matched_subgraph['onnx/Concat_10']]['data']
    ConstantOfShape_node = source_graph.nodes[matched_subgraph['onnx/ConstantOfShape_11']]['data']
    ConstantOfShape_shape = self.shape_dict[ConstantOfShape_node.name]
    ConstantOfShape_output = np.full(ConstantOfShape_shape, nh.to_array(ConstantOfShape_node.attribute[0].t)[0], nh.to_array(ConstantOfShape_node.attribute[0].t).dtype)
    mul_node = source_graph.nodes[matched_subgraph['onnx/Mul_12']]['data']
    mul_init = self.state_dict[mul_node.input[1]]
    result = ConstantOfShape_output * mul_init
    self.state_dict[mul_node.output[0]] = result
    for i in matched_subgraph.values():
      source_graph.remove_node(i)
    logger.info("Unet pattern 2 removed nodes: " + ','.join(i for i in matched_subgraph.values()))
    return None


class Unet3Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Shape_1', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_2', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_3', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_4', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape_5', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_6', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_7', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_8', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)

    node = onnx_make_dummy_node('onnx/Shape_9', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_10', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Shape_11', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_12', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_13', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_14', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Unsqueeze_15', 'Unsqueeze')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_16', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_17', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Concat_18', 'Concat')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Shape_1', 'onnx/Gather_2')))
    edge_list.append(tuple(('onnx/Gather_2', 'onnx/Unsqueeze_3')))
    edge_list.append(tuple(('onnx/Gather_2', 'onnx/Unsqueeze_4')))
    edge_list.append(tuple(('onnx/Shape_5', 'onnx/Gather_6')))
    edge_list.append(tuple(('onnx/Gather_6', 'onnx/Unsqueeze_7')))
    edge_list.append(tuple(('onnx/Gather_6', 'onnx/Unsqueeze_8')))

    edge_list.append(tuple(('onnx/Shape_9', 'onnx/Gather_10')))
    edge_list.append(tuple(('onnx/Shape_11', 'onnx/Gather_12')))
    edge_list.append(tuple(('onnx/Gather_10', 'onnx/Unsqueeze_13')))
    edge_list.append(tuple(('onnx/Gather_10', 'onnx/Mul_16')))
    edge_list.append(tuple(('onnx/Gather_12', 'onnx/Unsqueeze_14')))
    edge_list.append(tuple(('onnx/Gather_12', 'onnx/Mul_16')))

    edge_list.append(tuple(('onnx/Mul_16', 'onnx/Unsqueeze_15')))

    edge_list.append(tuple(('onnx/Unsqueeze_3', 'onnx/Concat_17')))
    edge_list.append(tuple(('onnx/Unsqueeze_13', 'onnx/Concat_17')))
    edge_list.append(tuple(('onnx/Unsqueeze_14', 'onnx/Concat_17')))
    edge_list.append(tuple(('onnx/Unsqueeze_7', 'onnx/Concat_17')))

    edge_list.append(tuple(('onnx/Unsqueeze_4', 'onnx/Concat_18')))
    edge_list.append(tuple(('onnx/Unsqueeze_15', 'onnx/Concat_18')))
    edge_list.append(tuple(('onnx/Unsqueeze_8', 'onnx/Concat_18')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape_1', 'onnx/Shape_5', 'onnx/Shape_9', 'onnx/Shape_11']

  def output_node(self):
    return "onnx/Concat_18"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    concat_1_node = source_graph.nodes[matched_subgraph['onnx/Concat_17']]['data']
    concat_2_node = source_graph.nodes[matched_subgraph['onnx/Concat_18']]['data']
    concat_1_shape = self.shape_dict[[i for i in source_graph.successors(concat_1_node.name)][0]]
    concat_2_shape = self.shape_dict[[i for i in source_graph.successors(concat_2_node.name)][0]]
    self.state_dict[concat_1_node.output[0]] = np.array(concat_1_shape)
    self.state_dict[concat_2_node.output[0]] = np.array(concat_2_shape)
    for i in matched_subgraph.values():
      source_graph.remove_node(i)
    logger.info("Unet pattern 3 removed nodes: " + ','.join(i for i in matched_subgraph.values()))
    return None


class Unet4Matcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()

    node = onnx_make_dummy_node('onnx/Shape_1', 'Shape')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Gather_2', 'Gather')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Add_3', 'Add')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Div_4', 'Div')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_5', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)
    node = onnx_make_dummy_node('onnx/Mul_6', 'Mul')
    G.add_node(node.name, data=node, type=node.op_type)

    edge_list.append(tuple(('onnx/Shape_1', 'onnx/Gather_2')))
    edge_list.append(tuple(('onnx/Gather_2', 'onnx/Add_3')))
    edge_list.append(tuple(('onnx/Add_3', 'onnx/Div_4')))
    edge_list.append(tuple(('onnx/Div_4', 'onnx/Mul_5')))
    edge_list.append(tuple(('onnx/Div_4', 'onnx/Mul_6')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/Shape_1']

  def output_node(self):
    return "onnx/Mul_6"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    shape_node = source_graph.nodes[matched_subgraph['onnx/Shape_1']]['data']
    shape = self.state_dict[shape_node.input[0]]
    gather_init = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/Gather_2']]['data'].input[1]]
    add_init = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/Add_3']]['data'].input[1]]
    div_init = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/Div_4']]['data'].input[1]]
    mul_1_init = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/Mul_5']]['data'].input[1]]
    mul_2_init = self.shape_dict[source_graph.nodes[matched_subgraph['onnx/Mul_6']]['data'].input[1]]
    if not all([i.size == 1 and i.shape == tuple([1]) for i in [gather_init, add_init, div_init, mul_1_init, mul_2_init]]):
      return None
    mul_1_result = ((shape[gather_init[0]] + add_init) / div_init).astype(div_init.dtype) * mul_1_init
    mul_2_result = ((shape[gather_init[0]] + add_init) / div_init).astype(div_init.dtype) * mul_2_init
    mul_1_node = source_graph.nodes[matched_subgraph['onnx/Mul_5']]['data']
    mul_2_node = source_graph.nodes[matched_subgraph['onnx/Mul_6']]['data']
    self.state_dict[mul_1_node.output[0]] = mul_1_result
    self.state_dict[mul_2_node.output[0]] = mul_2_result
    for i in matched_subgraph.values():
      source_graph.remove_node(i)
    logger.info("Unet pattern 4 removed nodes: " + ','.join(i for i in matched_subgraph.values()))
    return None


class RMSNormMatcher(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict
    self.counter = 0
    self.replacement_is_graph = True
    self.replacement_output_node = None

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    cast = onnx_make_dummy_node('onnx/cast', 'Cast')
    power = onnx_make_dummy_node('onnx/pow', 'Pow')
    rmean = onnx_make_dummy_node('onnx/rmean', 'ReduceMean')
    add = onnx_make_dummy_node('onnx/add', 'Add')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    for node in [cast, power, rmean, add, sqrt, div, mul]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/cast', 'onnx/pow')))
    edge_list.append(tuple(('onnx/pow', 'onnx/rmean')))
    edge_list.append(tuple(('onnx/rmean', 'onnx/add')))
    edge_list.append(tuple(('onnx/add', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/div')))
    edge_list.append(tuple(('onnx/div', 'onnx/mul')))
    edge_list.append(tuple(('onnx/cast', 'onnx/mul')))
    G.add_edges_from(edge_list)
    return G

  def subgraph_to_replace(self) -> nx.DiGraph:
    """Returns replacement subgraph template"""
    G = nx.DiGraph()
    edge_list = list()
    self.l2norm = 'rms_ara_l2norm'
    self.mul = 'rms_ara_mul'
    l2norm_node = onnx_make_dummy_node(self.l2norm, 'L2Norm')
    mul_node = onnx_make_dummy_node(self.mul, 'Mul')
    for node in [l2norm_node, mul_node]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple((self.l2norm, self.mul)))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    return ['onnx/cast']

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    return "onnx/mul"

  def get_replacement_output_node(self) -> str:
    """Returns the name of the output node in the replacement graph"""
    return self.replacement_output_node

  def get_input_to_replacement_graph_dict(self):
    """"Returns dictionary mapping node in replacement graph to corresponding node outside the replacement graph feeding input"""
    return self.input_node_to_replacement_graph_dict

  def mk_replacement_graph(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None) -> nx.DiGraph:
    """
    Constructs a replacement node
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: The matched subgraph
      model: ONNX Model
      inputs_to_collaphse: 
    Returns:
      replacement_graph: Replacement netx subgraph
    """
    logger.info("Entering RMSNorm Pattern matcher")
    _data = source_graph.nodes[matched_subgraph['onnx/pow']]['data']
    _add_data = source_graph.nodes[matched_subgraph['onnx/add']]['data']
    # if source_graph.nodes[matched_subgraph['onnx/pow']]['data'].input[0] != source_graph.nodes[matched_subgraph['onnx/div']]['data'].input[0]:
    # return None

    input_shape = self.shape_dict[_data.input[0]]
    power = self.state_dict[_data.input[1]]
    if (power != 2):
      return None

    replacement_graph = self.subgraph_to_replace()
    # Reusing source node name for output of replacement node
    self.l2norm = matched_subgraph['onnx/mul'] + '_ara_l2norm'
    self.mul = matched_subgraph['onnx/mul']
    name_mapping = {'rms_ara_l2norm': self.l2norm, 'rms_ara_mul': self.mul}
    replacement_graph = relabel_netx_graph(replacement_graph, name_mapping)
    # Replacing names inside graph and graph data

    replacement_graph.nodes[self.l2norm]['data'].input[:] = [source_graph.nodes[matched_subgraph['onnx/cast']]['data'].input[0]]
    replacement_graph.nodes[self.l2norm]['data'].attribute.append(helper.make_attribute("across_spatial", True))
    replacement_graph.nodes[self.l2norm]['data'].attribute.append(helper.make_attribute("per_channel", True))
    l2norm_out = self.l2norm + '_output'
    replacement_graph.nodes[self.l2norm]['data'].output[:] = [l2norm_out]
    if _data.input[0] in self.shape_dict:
      set_dict(self.shape_dict, l2norm_out, self.shape_dict[_data.input[0]])
    mul_constant = matched_subgraph['onnx/mul'] + '_sqrt_n_constant_input'
    mul_constant_val = np.sqrt([input_shape[-1]]).astype('float32')
    replacement_graph.nodes[self.l2norm]['data'].attribute.append(helper.make_attribute("eps", self.state_dict[_add_data.input[1]] * input_shape[-1]))
    mul_constant_tensor = np.full((input_shape[-1],), mul_constant_val, dtype='float32')
    set_dict(self.state_dict, mul_constant, mul_constant_tensor)
    replacement_graph.nodes[self.mul]['data'].input[:] = [l2norm_out, mul_constant]
    replacement_graph.nodes[self.mul]['data'].output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    #Set output node
    self.replacement_output_node = self.mul
    self.input_node_to_replacement_graph_dict = {self.l2norm: list(source_graph.predecessors(matched_subgraph['onnx/cast']))[0]}
    logger.info("RMSNorm pattern replaced")
    return replacement_graph



class RMSNormMatcher2(BaseMatcher):

  def __init__(self, model, state_dict, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    power = onnx_make_dummy_node('onnx/pow', 'Pow')
    rmean = onnx_make_dummy_node('onnx/rmean', 'ReduceMean')
    add = onnx_make_dummy_node('onnx/add', 'Add')
    sqrt = onnx_make_dummy_node('onnx/sqrt', 'Sqrt')
    div = onnx_make_dummy_node('onnx/div', 'Div')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    mul_2 = onnx_make_dummy_node('onnx/mul_2', 'Mul')
    for node in [power, rmean, add, sqrt, div, mul, mul_2]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/pow', 'onnx/rmean')))
    edge_list.append(tuple(('onnx/rmean', 'onnx/add')))
    edge_list.append(tuple(('onnx/add', 'onnx/sqrt')))
    edge_list.append(tuple(('onnx/sqrt', 'onnx/div')))
    edge_list.append(tuple(('onnx/div', 'onnx/mul')))
    edge_list.append(tuple(('onnx/mul', 'onnx/mul_2')))
    G.add_edges_from(edge_list)
    return G


  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    return ['onnx/pow']

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    return "onnx/mul_2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("Entering RMSNorm Pattern matcher2")
    pow_input = source_graph.nodes[matched_subgraph['onnx/pow']]['data'].input[0]
    div_output = source_graph.nodes[matched_subgraph['onnx/div']]['data'].output[0]
    mul_input2 = [pow_input, div_output] 
    mul_input = set(source_graph.nodes[matched_subgraph['onnx/mul']]['data'].input)
    if set(mul_input2) != mul_input:
      return None
    _data = source_graph.nodes[matched_subgraph['onnx/pow']]['data']
    eps = self.state_dict[source_graph.nodes[matched_subgraph['onnx/add']]['data'].input[1]].tolist()
    gamma = self.state_dict[source_graph.nodes[matched_subgraph['onnx/mul_2']]['data'].input[0]]
    if len(source_graph.nodes[matched_subgraph['onnx/rmean']]['data'].attribute[0].ints) != 1:
      return None
    axes1 = source_graph.nodes[matched_subgraph['onnx/rmean']]['data'].attribute[0].ints[0]
    output_mean_var = False
    uses_mean = False
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "LayerNorm")
    node_def_attr = helper.make_attribute("eps", eps)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("axes", axes1)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("output_mean_var", output_mean_var)
    node_def.attribute.append(node_def_attr)
    node_def_attr = helper.make_attribute("uses_mean", uses_mean)
    node_def.attribute.append(node_def_attr)
    shape = self.shape_dict[source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output[0]]
    set_dict(self.state_dict,matched_subgraph[self.output_node()]+"_scale",gamma)
    set_dict(self.state_dict,matched_subgraph[self.output_node()]+"_bias",np.zeros(shape[axes1]))
    node_def.input[:] = [source_graph.nodes[matched_subgraph['onnx/pow']]['data'].input[0],matched_subgraph[self.output_node()]+"_scale",matched_subgraph[self.output_node()]+"_bias"]
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    logger.info("leaving RMSNorm Pattern matcher2")
    return node_def
    


class RopeMatcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()

    slice1 = onnx_make_dummy_node('onnx/slice1', 'Slice')
    slice2 = onnx_make_dummy_node('onnx/slice2', 'Slice')
    neg = onnx_make_dummy_node('onnx/neg', 'Neg')
    concat = onnx_make_dummy_node('onnx/concat', 'Concat')
    mul1 = onnx_make_dummy_node('onnx/mul1', 'Mul')
    mul2 = onnx_make_dummy_node('onnx/mul2', 'Mul')
    add = onnx_make_dummy_node('onnx/add', 'Add')
    
    for node in [slice1, slice2, neg, concat, mul1, mul2, add]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/slice1', 'onnx/concat')))
    edge_list.append(tuple(('onnx/slice2', 'onnx/neg')))
    edge_list.append(tuple(('onnx/neg', 'onnx/concat')))
    edge_list.append(tuple(('onnx/concat', 'onnx/mul2')))
    edge_list.append(tuple(('onnx/mul1', 'onnx/add')))
    edge_list.append(tuple(('onnx/mul2', 'onnx/add')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/mul1', 'onnx/slice1', 'onnx/slice2', 'onnx/mul2']

  def output_node(self):
    return "onnx/add"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("replacing with Rope")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Rope")
    input0 = source_graph.nodes[matched_subgraph['onnx/slice1']]['data'].input[0]
    input1 = source_graph.nodes[matched_subgraph['onnx/mul1']]['data'].input[1]
    input2 = source_graph.nodes[matched_subgraph['onnx/mul2']]['data'].input[1]
    #keep inputs in order
    node_def.input[:] = [input0, input1] + list(set(inputs_to_collapsed) - {input0, input1, input2})
    node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
    return node_def

class RopeMulMatcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()

    rope = onnx_make_dummy_node('onnx/rope', 'Rope')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    
    for node in [rope, mul]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    edge_list.append(tuple(('onnx/rope', 'onnx/mul')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/rope']

  def output_node(self):
    return "onnx/mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.info("removing Mul after Rope")
    _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Rope")
    input0 = source_graph.nodes[matched_subgraph['onnx/rope']]['data'].input[0]
    input1 = source_graph.nodes[matched_subgraph['onnx/rope']]['data'].input[1]

    node_def.input[:] = inputs_to_collapsed
    node_def.output[:] = source_graph.nodes[matched_subgraph['onnx/mul']]['data'].output
    return node_def

class RopeTransposeMulMatcher(BaseMatcher):

  def __init__(self, model):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()

  def subgraph_to_match(self):
    G = nx.DiGraph()

    # rope = onnx_make_dummy_node('onnx/rope', 'Rope')
    trasnspose = onnx_make_dummy_node('onnx/transpose', 'Transpose')
    mul = onnx_make_dummy_node('onnx/mul', 'Mul')
    
    for node in [mul, trasnspose]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list = []
    # edge_list.append(tuple(('onnx/rope', 'onnx/transpose')))
    edge_list.append(tuple(('onnx/transpose', 'onnx/mul')))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['onnx/transpose']

  def output_node(self):
    return "onnx/mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    if (source_graph.nodes[[i for i in source_graph.predecessors(matched_subgraph['onnx/transpose'])][0]]['data'].op_type == 'Rope'):      
      logger.info("removing Mul after [Rope->Transpose]")
      _data = source_graph.nodes[matched_subgraph['onnx/transpose']]['data']
      node_def = onnx_make_dummy_node(matched_subgraph[self.output_node()], "Transpose")
      _proto_copy_attribute(_data, node_def)
      node_def.input[:] = inputs_to_collapsed
      node_def.output[:] = source_graph.nodes[matched_subgraph[self.output_node()]]['data'].output
      return node_def
    return None
  
class UnsqueezeAfterSinCos_Matcher(BaseMatcher):

  def __init__(self, model, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.shape_dict = shape_dict
    self.counter = 0
    self.replacement_is_graph = True
    self.replacement_output_node = None

  def subgraph_to_match(self) -> nx.DiGraph:
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    cos = onnx_make_dummy_node('onnx_cos', 'Cos')
    sin = onnx_make_dummy_node('onnx_sin', 'Sin')
    unsqueeze1 = onnx_make_dummy_node('onnx_unsqueeze1', 'Unsqueeze')
    unsqueeze2 = onnx_make_dummy_node('onnx_unsqueeze2', 'Unsqueeze')
    #breakpoint()

    for node in [cos, sin, unsqueeze1, unsqueeze2]:
      G.add_node(node.name, data=node, type=node.op_type)
    
    edge_list.append(tuple(('onnx_cos', 'onnx_unsqueeze1')))
    edge_list.append(tuple(('onnx_sin', 'onnx_unsqueeze2')))
    G.add_edges_from(edge_list)

    return G

  def subgraph_to_replace(self) -> nx.DiGraph:
    """Returns replacement subgraph template"""
    G = nx.DiGraph()
    edge_list = list()
    self.cos = 'onnx_cos'
    self.sin = 'onnx_sin'
    self.unsqueeze1 = 'onnx_unsqueeze1'
    self.unsqueeze2 = 'onnx_unsqueeze2'
    
    cos = onnx_make_dummy_node(self.cos, 'Cos')
    sin = onnx_make_dummy_node(self.sin, 'Sin')
    unsqueeze1 = onnx_make_dummy_node(self.unsqueeze1, 'Unsqueeze')
    unsqueeze2 = onnx_make_dummy_node(self.unsqueeze2, 'Unsqueeze')
    #breakpoint()
    self.new_concat1 = 'position_ids_embedded'
    concat1_node = onnx_make_dummy_node(self.new_concat1, 'Concat')

    for node in [cos, sin, unsqueeze1, unsqueeze2, concat1_node]:
      G.add_node(node.name, data=node, type=node.op_type)
    edge_list.append(tuple(('onnx_cos', 'onnx_unsqueeze1')))
    edge_list.append(tuple(('onnx_sin', 'onnx_unsqueeze2')))
    edge_list.append(tuple(('onnx_unsqueeze1', self.new_concat1)))
    edge_list.append(tuple(('onnx_unsqueeze2', self.new_concat1)))

    G.add_edges_from(edge_list)
    return G

  def input_nodes(self) -> List[str]:
    """Returns names of nodes receiving input in the template subgraph to match"""
    return ['onnx_cos', 'onnx_sin']

  def output_node(self) -> str:
    """Returns name of output node in template subgraph to match"""
    return "onnx_unsqueeze1 onnx_unsqueeze2"

  def get_replacement_output_node(self) -> str:
    """Returns the name of the output node in the replacement graph"""
    return self.replacement_output_node

  def get_input_to_replacement_graph_dict(self):
    """"Returns dictionary mapping node in replacement graph to corresponding node outside the replacement graph feeding input"""
    return self.input_node_to_replacement_graph_dict

  def mk_replacement_graph(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None) -> nx.DiGraph:
    """
    Constructs a replacement node
    Args:
      source_graph: The netx graph of original model
      matched_subgraph: The matched subgraph
      model: ONNX Model
      inputs_to_collaphse: 
    Returns:
      replacement_graph: Replacement netx subgraph
    """
    # _data = source_graph.nodes[matched_subgraph[self.output_node()]]['data']
    # _axis = _data.attribute[0].i
    # if (len(_data.input) != 4):
    #   return None
    replacement_graph = self.subgraph_to_replace()
    # # Reusing source node name for output of replacement node
    # self.new_concat3 = matched_subgraph['onnx/concat']
    # name_mapping = {'concat_ara3': self.new_concat3}
    # idx = self.update_counter()
    # if idx > 0:  #In case there are multiple matches, pass all the layer names that would name renaming
    #   [self.new_concat1, self.new_concat2] = rename_layers([self.new_concat1, self.new_concat2], name_mapping, idx)
    # # Replacing names inside graph and graph data
    # replacement_graph = relabel_netx_graph(replacement_graph, name_mapping)

    # # Setting inputs, outputs and tensor shapes
    _data = source_graph.nodes[matched_subgraph[self.cos]]['data']
    replacement_graph.nodes[self.cos]['data'].input[:] = _data.input
    replacement_graph.nodes[self.cos]['data'].output[:] = [self.cos]
    set_dict(self.shape_dict, replacement_graph.nodes[self.cos]['data'].output[0], self.shape_dict[_data.output[0]])

    _data = source_graph.nodes[matched_subgraph[self.sin]]['data']
    replacement_graph.nodes[self.sin]['data'].input[:] = _data.input
    replacement_graph.nodes[self.sin]['data'].output[:] = [self.sin]
    set_dict(self.shape_dict, replacement_graph.nodes[self.sin]['data'].output[0], self.shape_dict[_data.output[0]])

    _data_unsqueeze1 = source_graph.nodes[matched_subgraph[self.unsqueeze1]]['data']
    replacement_graph.nodes[self.unsqueeze1]['data'].input[:] = [self.cos]
    replacement_graph.nodes[self.unsqueeze1]['data'].output[:] = [self.unsqueeze1]
    set_dict(self.shape_dict, replacement_graph.nodes[self.unsqueeze1]['data'].output[0], self.shape_dict[_data_unsqueeze1.output[0]])

    _data_unsqueeze2 = source_graph.nodes[matched_subgraph[self.unsqueeze2]]['data']
    replacement_graph.nodes[self.unsqueeze2]['data'].input[:] = [self.sin]
    replacement_graph.nodes[self.unsqueeze2]['data'].output[:] = [self.unsqueeze2]
    set_dict(self.shape_dict, replacement_graph.nodes[self.unsqueeze2]['data'].output[0], self.shape_dict[_data_unsqueeze2.output[0]])
    
    _axis = -1
    old_shape = deepcopy(self.shape_dict[_data_unsqueeze1.output[0]])
    old_shape[_axis] *= 2
    new_shape_1 = deepcopy(old_shape)
    #breakpoint()
    # replacement_graph.nodes[self.new_concat1]['data'].input[:] = [self.unsqueeze1, self.unsqueeze2]
    replacement_graph.nodes[self.new_concat1]['data'].input.MergeFrom([self.unsqueeze1, self.unsqueeze2])
    replacement_graph.nodes[self.new_concat1]['data'].output[:] = [self.new_concat1]
    add_attribute(replacement_graph.nodes[self.new_concat1]['data'], "axis", _axis)
    set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat1]['data'].output[0], new_shape_1)
    # old_shape = deepcopy(self.shape_dict[_data.input[0]])
    # old_shape[_axis] *= 2
    # new_shape_1 = deepcopy(old_shape)
    # set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat1]['data'].output[0], new_shape_1)

    # replacement_graph.nodes[self.new_concat2]['data'].input[:] = _data.input[2:4]
    # replacement_graph.nodes[self.new_concat2]['data'].output[:] = [self.new_concat2]
    # add_attribute(replacement_graph.nodes[self.new_concat2]['data'], "axis", _axis)
    # set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat2]['data'].output[0], new_shape_1)

    # replacement_graph.nodes[self.new_concat3]['data'].input[:] = [self.new_concat1, self.new_concat2]
    # replacement_graph.nodes[self.new_concat3]['data'].output[:] = [self.new_concat3]
    # add_attribute(replacement_graph.nodes[self.new_concat3]['data'], "axis", _axis)
    # old_shape[_axis] *= 2
    # new_shape_2 = deepcopy(old_shape)
    # set_dict(self.shape_dict, replacement_graph.nodes[self.new_concat3]['data'].output[0], new_shape_2)

    # #Set output node
    self.replacement_output_node = self.new_concat1
    self.input_node_to_replacement_graph_dict = {self.cos: list(source_graph.predecessors(matched_subgraph[self.cos]))[0], self.sin: list(source_graph.predecessors(matched_subgraph[self.sin]))[0]}
    #breakpoint()
    return replacement_graph


class llama_unsqueeze_gather_Matcher(BaseMatcher):
  def __init__(self, model, shape_dict):
    self.tensor_to_name_map = _output_tensor_to_name_map(model)
    self.value_info = dict()
    self.shape_dict = shape_dict

  def subgraph_to_match(self):
    """Returns template of subgraph to match"""
    G = nx.DiGraph()
    edge_list = list()
    cos = onnx_make_dummy_node('onnx_cos', 'Gather')
    sin = onnx_make_dummy_node('onnx_sin', 'Gather')
    unsqueeze1 = onnx_make_dummy_node('onnx_unsqueeze1', 'Unsqueeze')
    unsqueeze2 = onnx_make_dummy_node('onnx_unsqueeze2', 'Unsqueeze')
    #breakpoint()

    for node in [cos, sin, unsqueeze1, unsqueeze2]:
      G.add_node(node.name, data=node, type=node.op_type)
    
    edge_list.append(tuple(('onnx_cos', 'onnx_unsqueeze1')))
    edge_list.append(tuple(('onnx_sin', 'onnx_unsqueeze2')))
    G.add_edges_from(edge_list)

    return G

  def input_nodes(self):
    return ['onnx_cos', 'onnx_sin']

  def output_node(self):
    return "onnx_unsqueeze1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    concat_new_node = onnx_make_dummy_node('position_ids_embedded', "Concat")
    unsqueeze1_node = source_graph.nodes[matched_subgraph['onnx_unsqueeze1']]['data']
    unsqueeze2_node = source_graph.nodes[matched_subgraph['onnx_unsqueeze2']]['data']
    concat_new_node.input[:] = [unsqueeze1_node.output[0], unsqueeze2_node.output[0]]
    unsqueeze1_succs = [source_graph.nodes[i]['data'] for i in source_graph.successors(unsqueeze1_node.name)]
    unsqueeze2_succs = [source_graph.nodes[i]['data'] for i in source_graph.successors(unsqueeze2_node.name)]
    new_edge_list, old_edge_list = [], []
    for i in unsqueeze1_succs:
      for j in range(len(i.input)):
        if i.input[j] == unsqueeze1_node.output[0]:
          i.input[j] = concat_new_node.output[0]
      new_edge_list.append(tuple((concat_new_node.name, i.name)))
      old_edge_list.append(tuple((unsqueeze1_node.name, i.name)))
    for i in unsqueeze2_succs:
      for j in range(len(i.input)):
        if i.input[j] == unsqueeze2_node.output[0]:
          i.input[j] = concat_new_node.output[0]
      new_edge_list.append(tuple((concat_new_node.name, i.name)))
      old_edge_list.append(tuple((unsqueeze2_node.name, i.name)))
    new_edge_list.append(tuple((unsqueeze1_node.name, concat_new_node.name)))
    new_edge_list.append(tuple((unsqueeze2_node.name, concat_new_node.name)))
    concat_new_node.attribute.append(helper.make_attribute("axis", 3))
    source_graph.add_node(concat_new_node.name, data=concat_new_node, type=concat_new_node.op_type)
    source_graph.add_edges_from(new_edge_list)
    source_graph.remove_edges_from(old_edge_list)
    self.shape_dict[concat_new_node.output[0]] = self.shape_dict[unsqueeze1_node.output[0]]
    return None


''' ! Reference 
def onnx_subGraph_match_replace_Test(G, model):
    return TestMatcher().replace_subgraph_instances(G, model)
'''


def onnx_subGraph_match_replace_PriorBoxDotMatcher(G, model, path, shape_dict, state_dict):
  return PriorBoxDotMatcher(path, shape_dict, state_dict).get_priorbox_outputs(G, model)


def onnx_subGraph_match_replace_L2NormMatcher(G, model):
  return L2NormMatcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_zt(G, model):
  return L2NormMatcher_zt(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_AC(G, model, shape_dict, state_dict):
  return L2NormMatcher_AC(model, shape_dict, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L1NormMatcher(G, model, state_dict, shape_dict):
  return L1NormMatcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_DetectionBoxMatcher(G, model, path, shape_dict):
  return DetectionBoxMatcher(path, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Centernet_IR1_Matcher(G, model):
  return Centernet_IR1_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ReduceSum_To_ELTS3_Matcher(G, model):
  return ReduceSum_To_ELTS3_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ReduceSum_To_ELTS2_Matcher(G, model):
  return ReduceSum_To_ELTS2_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_PytorchOnnxSoftmax_PSP_Matcher(G, model):
  return PytorchOnnxSoftmax_PSP_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_PytorchOnnxSoftmax_Matcher(G, model):
  return PytorchOnnxSoftmax_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Centernet_IR4_Matcher(G, model):
  return Centernet_IR4_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Swish_Matcher(G, model):
  return Swish_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Mish_Matcher(G, model):
  return Mish_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_yolov8_scatterND_Matcher(G, model, state_dict, shape_dict):
  return Yolov8_scatterND_Replacer(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_testGraph_Matcher(G, model):
  logger.info("Entering test graph matcher")
  return TestGraphReplacer(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_concat_Matcher(G, model, shape_dict):
  logger.info("Entering concat graph matcher")
  return TestConcat_Matcher(model, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Tanh_Matcher(G, model):
  return Tanh_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Yolo4_Expand_To_Upsample_Matcher(G, model):
  return Yolo4_Expand_To_Upsample_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_simba(G, model):
  return L2NormMatcher_simba(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_simba_1(G, model):
  return L2NormMatcher_simba_1(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_clip(G, model):
  return L2NormMatcher_clip(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Advertima_IR(G, model):
  return Advertima_IR(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_1(G, model):
  return L2NormMatcher_1(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Convnext_simplify(G, model, state_dict, shape_dict):
  return Convnext_simplify_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_LayerNormMatcher_1(G, model):
  return LayerNormMatcher_1(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_LayerNormMatcher_2(G, model):
  return LayerNormMatcher_2(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_LayerNormMatcher_3(G, model):
  return LayerNormMatcher_3(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_LayerNormMatcher_4(G, model):
  return LayerNormMatcher_4(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_RoiAlignMax_Matcher(G, model, state_dict):
  return RoiAlignMax_Matcher(model, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_FocusLayer_Matcher(G, model):
  return FocusLayer_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_FocusLayer2_Matcher(G, model):
  return FocusLayer2_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GeluLayer_Matcher(G, model):
  return GeluLayer_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GeluLayer2_Matcher(G, model):
  return GeluLayer2_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GeluLayer3_Matcher(G, model, state_dict, shape_dict):
  return GeluLayer3_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_LayerNormP1_Matcher(G, model, state_dict):
  return LayerNormP1_Matcher(model, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_LayerNormP2_Matcher(G, model, state_dict):
  return LayerNormP2_Matcher(model, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_SFRoiMaxLayer_Matcher(G, model):
  return SFRoiMaxLayer_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_SFRoiLayer_Matcher(G, model):
  return SFRoiLayer_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_SFRoi2Layer_Matcher(G, model):
  return SFRoi2Layer_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_remove_CastReshape1(G, model):
  return CastReshape1_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_remove_CastReshape2(G, model):
  return CastReshape2_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ReshapeUnsqueeze(G, model, shape_dict):
  return ReshapeUnsqueeze_Matcher(model, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_FullyConnected(G, model, state_dict, shape_dict):
  return FullyConnected_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_MatMul(G, model, state_dict, shape_dict):
  return Matmul_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConcatUpsample_Matcher(G, model):
  return ConcatUpsample_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_DensePose_upsample_Matcher(G, model):
  return DensePose_Upsample_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_L2NormMatcher_2(G, model):
  return L2NormMatcher_2(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_AliaswithNameRoi_Matcher(G, model):
  return AliaswithNameRoi_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GCD_Matcher(G, model):
  return GCD_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GroupnormCast_Matcher(G, model, state_dict, shape_dict):
  return GroupnormCast_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_WeedOut_Matcher(G, model):
  return WeedOut(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_IBN(G, model, state_dict):
  return IBNMatcher(model, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_RepeatInterleave_Matcher(G, model):
  return RepeatInterleave_Matcher(model).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GroupNormMatcher(G, model, state_dict, shape_dict):
  return GroupNorm_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_GroupNorm1Matcher(G, model, state_dict, shape_dict):
  return GroupNorm_1_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Tile_Mul_Matcher(G, model, state_dict, shape_dict):
  return Tile_Mul_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Flynn_d_tail_Matcher(G, model, state_dict, shape_dict):
  return Flynn_d_tail_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_BBoxTransformDotMatcher(G, model, path, shape_dict, state_dict):
  return BBoxTransformDotMatcher(path, shape_dict, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_BBoxTransformDotStaticMatcher(G, model, path, shape_dict, state_dict):
  return BBoxTransformDotStaticMatcher(path, shape_dict, state_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Mul2_Matcher(G, model, state_dict, shape_dict):
  return Mul2_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_NullPattern_Matcher(G, model, state_dict, shape_dict):
  return NullPattern_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstCast_Matcher(G, model, state_dict, shape_dict):
  return ConstCast_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstPow_Matcher(G, model, state_dict, shape_dict):
  return ConstPow_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstUnsqueeze_Matcher(G, model, state_dict, shape_dict):
  return ConstUnsqueeze_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstDiv_Matcher(G, model, state_dict, shape_dict):
  return ConstDiv_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstGather_Matcher(G, model, state_dict, shape_dict):
  return ConstGather_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstReshape_Matcher(G, model, state_dict, shape_dict):
  return ConstReshape_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ConstAdd_Matcher(G, model, state_dict, shape_dict):
  return ConstAdd_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_ZeroTensorAdd_Matcher(G, model, state_dict, shape_dict):
  return ZeroTensorAdd_Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Unet_1(G, model, shape_dict, state_dict):
  return Unet1Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Unet_2(G, model, shape_dict, state_dict):
  return Unet2Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Unet_3(G, model, shape_dict, state_dict):
  return Unet3Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_match_replace_Unet_4(G, model, shape_dict, state_dict):
  return Unet4Matcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)


def onnx_subGraph_math_replace_RMSNormMatcher(G, model, state_dict, shape_dict):
  return RMSNormMatcher(model, state_dict, shape_dict).replace_subgraph_instances(G, model)

def onnx_subGraph_match_replace_RMSNormMatcher2(G, model, state_dict, shape_dict):
  return RMSNormMatcher2(model, state_dict, shape_dict).replace_subgraph_instances(G, model)

def onnx_subGraph_match_replace_Rope_Matcher(G, model):
  return RopeMatcher(model).replace_subgraph_instances(G, model)

def onnx_subGraph_match_replace_RopeMul_Matcher(G, model):
  return RopeMulMatcher(model).replace_subgraph_instances(G, model)

def onnx_subGraph_match_replace_RopeTransposeMul_Matcher(G, model):
  return RopeTransposeMulMatcher(model).replace_subgraph_instances(G, model)

def onnx_subGraph_match_replace_UnsqueezeAfterSinCos_Matcher_Matcher(G, model, shape_dict):
  return UnsqueezeAfterSinCos_Matcher(model, shape_dict).replace_subgraph_instances(G, model)

def onnx_subGraph_match_replace_llama_unsqueeze_gather_Matcher(G, model, shape_dict):
  return llama_unsqueeze_gather_Matcher(model, shape_dict).replace_subgraph_instances(G, model)