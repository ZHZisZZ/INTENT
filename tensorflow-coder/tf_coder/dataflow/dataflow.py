from typing import Dict, List, Tuple, Text, Any
from collections import defaultdict
import tensorflow as tf

from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search as value_search_module
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.dataflow import dataflow_utils, trace


def value_from_different_inputs(
    value: value_module.OperationValue, 
    inputs: List[Any]
) -> value_module.OperationValue:
  """Rebuild the same expression but from different inputs"""
  
  if isinstance(value, value_module.ConstantValue):
    return value
  
  if isinstance(value, value_module.InputValue):
    input_names_to_objects = \
        value_search_module._input_names_to_objects(inputs)
    return value_module.InputValue(input_names_to_objects[value.name], value.name)
  
  operation, arg_values = value.operation_applications[0]
  
  arg_values = [value_from_different_inputs(arg, inputs) for arg in arg_values]
  
  return operation.apply(arg_values, settings_module.default_settings())


def value_from_text(
  expression: Text,
  inputs: List[Any]
) -> value_module.OperationValue:
  # parse structure
  parsed_struct = dataflow_utils.parse_expression(expression)
  
  # not tensor function
  if not parsed_struct.operation_name:
    
    input_names_to_objects = \
        value_search_module._input_names_to_objects(inputs)
    # input
    if expression in input_names_to_objects.keys():
      return value_module.InputValue(input_names_to_objects[expression], expression)
    # constant
    namespace = {'tf': tf}
    return value_module.ConstantValue(eval(expression, namespace))
  
  operation = dataflow_utils.find_operation_with_parsed_struct(parsed_struct)
  
  arg_values = [value_from_text(arg, inputs) for arg in parsed_struct.list_of_args]
    
  return operation.apply(arg_values, settings_module.default_settings())
  

def dataflow_generator(
    value: value_module.OutputValue
) -> Tuple[Dict[Text, Dict[Text, Any]],         # value nodes
           List[Dict[Text, Any]],               # edges
           Dict[Text, List[Dict[Text, Any]]]]:  # trace edges
  """Return (nodes, edges) as in dataflow diagram.

  The same intermediate values are merged with each other;
  The same constant values are merged with each other;
  The input values are not merged even if they have the same values,
    i.e. in1 = [1], in2 = [1]; But if one input is used multiple times,
    the input node is not copied, i.e. tf.add(in1, in1), where we only
    have one 'in1' node and two edges feeding into 'tf.add' operator.
  """
  # # a dict that maps value to node key 'n*'
  # value_to_value_node_key = {}  # Dict[Text, Text]
  
  # expression -> parent node -> node key 'n*'
  value_node_key_map = defaultdict(defaultdict)
  
  value_node_cnt = 0

  # a dict that maps node key 'n*' to node info
  value_nodes = {}  # Dict[Text(n*), Dict[Text, Any]]

  # a dict that maps operation key 'p*' to opertaion info
  op_nodes = {}  # Dict[Text(p*), Dict[Text, Any]]

  # a list of dict that contains start and end node/operation key
  edges = []  # List[Dict[Text, Text]]

  trace_edges = defaultdict(list)  # [Dict[Text, List[Dict[Text, Text]]

  class_to_str = {value_module.OperationValue: 'intermediate', 
                  value_module.ConstantValue: 'constant',
                  value_module.InputValue: 'input'}


  def _dataflow_generator(value: value_module, parent_node_key=None):
    """Recursive helper function."""
    expression = value.reconstruct_expression()

    # TODO: uncomment to merge nodes
    # if (expression in value_node_key_map and
    #     parent_node_key not in value_node_key_map[expression]):
    #   # arbitrary choose one
    #   value_node_key = list(value_node_key_map[expression].values())[0]
    #   value_node_key_map[expression][parent_node_key] = value_node_key
    #   return value_node_key

    # create node key 
    nonlocal value_node_cnt
    value_node_key = f'n{value_node_cnt}'
    value_node_cnt += 1

    # record the node key in dict
    value_node_key_map[expression][parent_node_key] = value_node_key

    # record node info
    value_nodes[value_node_key] = \
        { "type": class_to_str[value.__class__], 
          "value": dataflow_utils.object_to_string(value.value),
          "label": expression }
    
    if not isinstance(value, value_module.OperationValue):
      return value_node_key

    operation, arg_values = value.operation_applications[0]
    
    # create operation key
    op_key = f'p{len(op_nodes)}'
    
    # record operation info
    op_nodes[op_key] = {
      "value": operation.name, 
      "docstring": dataflow_utils.get_docstring(operation)}
    
    # op -> out edge
    edges.append({"start": op_key, "end": value_node_key})
    
    # trace
    trace_solutions = trace.trace(value)
    
    for arg_value, arg_name in zip(arg_values, 
                                   dataflow_utils.separate_arg_names(operation)):
      child_node_key = _dataflow_generator(arg_value, value_node_key)
      
      # in -> op edges
      edges.append({
        "start": child_node_key,
        "end": op_key, 
        "label": arg_name})
      
      if trace_solutions:
        trace_solution = trace_solutions[arg_name]
        trace_edges[value_node_key].append({
            'start': child_node_key,
            'is_value_wise': trace_solution.is_value_wise, 
            'description': trace_solution.description,
            'provenance': repr(trace_solution.provenance)})
        
    return value_node_key


  _dataflow_generator(value)
  value_nodes.update(op_nodes)
  return value_nodes, edges, trace_edges


if __name__ == '__main__':
  import json
  from tf_coder.benchmarks import all_benchmarks
  
  # benchmark = all_benchmarks.find_benchmark_with_name('google_11')
  # expression = benchmark.target_program
  # inputs = benchmark.examples[0].inputs
  
  value = value_from_text('(in1[:,1], in1)', [[[1,2],[3,4]]])
  n, e, te = dataflow_generator(value)
  print(json.dumps({'nodes': n, 'edges': e, 'trace_edges': te}, indent=2))
  print(value.value)

               