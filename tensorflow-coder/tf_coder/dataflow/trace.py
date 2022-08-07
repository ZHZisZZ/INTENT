import tensorflow as tf
import functools
import itertools
import copy
from typing import NamedTuple, List, Text, Dict, Tuple, Callable

from tf_coder.value_search import all_operations
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.dataflow import tf_functions, dataflow_utils

TraceSolution = NamedTuple('TraceSolution', [
    ('provenance', List),
    ('description', Text),
    ('is_value_wise', bool)
])


################################################################################
# Helper Functions.
def id_where(tensor: tf.Tensor):
  """Squeeze the tensor and then return a list of indices
  
  For example, if tensor=[[True,False],[False,True]], the function 
  should return [0, 3].
  """
  return tf.reshape(tf.where(tf.reshape(tensor, [-1,])), [-1,]).numpy().tolist()


def tensor_to_list(tensor: tf.Tensor, out_shape: Tuple, index_to_id: bool = True):
  """Convert a tensorflow tensor to python list.
  
  Args:
    tensor: A tensor with shape (a,...,b,m,...,n), where (a,...,b) is the 
      output tensor shape and (m,...,n) is the input tensor shape.
    out_shape: A tuple that represents output tensor shape, i.e., (a,...,b).

  Return:
    A python list with shape (a,...,b,len(m,...,n)). res[a,...,b] shows the 
      index of its provenance in the input tensor.

  If input tensor shape is empty, i.e., input tensor is a primitive or a tensor
    that wraps a primitive, res[a,...,b] shows True or False.
  """
  shape = tensor.shape

  # the shape of input tensor
  in_shape = shape[len(out_shape):]
  # if input tensor is a primitive, wraps it in a tensor
  if not len(in_shape):
    in_shape = [1,]

  # tensor with output dimension squeezed: (out_shape, in_shape) -> (-1, inshape)
  squeezed_tensor = tf.reshape(
      tensor, shape=[-1,] + list(in_shape))

  if not index_to_id:
    # convert each inner tensor with shape in_shape to a index list
    index_list = [tf.where(in_tensor).numpy().tolist()
        for in_tensor in squeezed_tensor]

    # reshape the index list to have shape out_shape
    return list(functools.reduce(
        lambda x, y: map(list, zip(*y*(x,))), 
        (iter(index_list), *out_shape[:0:-1])))

  else:
    return [id_where(in_tensor) for in_tensor in squeezed_tensor]


def parse_valuewise_args(value, func_info, trace_info):
  if tf_functions.ArgGroup.VALUEWISE not in func_info.arg_groups:
    return 

  operation, arg_values = value.operation_applications[0]

  for arg_value, arg_name, arg_group, description \
      in zip(arg_values, 
             dataflow_utils.separate_arg_names(operation),
             func_info.arg_groups, 
             func_info.description):

    if arg_group is not tf_functions.ArgGroup.VALUEWISE: 
        continue

    trace_info[arg_name]= \
        TraceSolution(None, description, True)  


def parse_all_edges_args(value, func_info, trace_info):
  if tf_functions.ArgGroup.ALL_EDGES not in func_info.arg_groups:
    return 

  operation, arg_values = value.operation_applications[0]

  for arg_value, arg_name, arg_group, description \
      in zip(arg_values, 
             dataflow_utils.separate_arg_names(operation),
             func_info.arg_groups, 
             func_info.description):

    if arg_group is not tf_functions.ArgGroup.ALL_EDGES: 
        continue

    value_shape = value.shape

    if arg_value.is_dtype:
      arg_value_shape = []
    else:
      arg_value_shape = tf.constant(arg_value.value).shape

    squeezed_value_shape = int(tf.reduce_prod(value_shape))
    arg_value_id_range = range(int(tf.reduce_prod(arg_value_shape)))

    provenance = [[i for i in arg_value_id_range]] * squeezed_value_shape

    trace_info[arg_name]= \
        TraceSolution(provenance, description, False)


def parse_elementwise_args(value, func_info, trace_info):
  if tf_functions.ArgGroup.ELEMENTWISE not in func_info.arg_groups:
    return 

  operation, arg_values = value.operation_applications[0]

  for arg_value, arg_name, arg_group, description \
      in zip(arg_values, 
             dataflow_utils.separate_arg_names(operation),
             func_info.arg_groups, 
             func_info.description):

    if arg_group is not tf_functions.ArgGroup.ELEMENTWISE: continue

    # `value` is guaranteed to be a tensor in elementwise operation
    value_shape = value.shape
    # in case `arg_value` is a python primitive or a sequence of tensor 
    # (where tf.constant fails)
    # arg_value_shape = tf.constant(arg_value.value).shape
    arg_value_shape = tf.cast(arg_value.value, tf.int32).shape

    assert value_shape == arg_value_shape, \
        f'{value}.shape {value_shape} != {arg_value}.shape {arg_value_shape}'

    id_range = range(int(tf.reduce_prod(value_shape)))
    provenance = [[i] for i in id_range]

    trace_info[arg_name]= \
        TraceSolution(provenance, description, False)


def parse_differentiable_args(value, func_info, trace_info):
  # assume all arg_value are tensors
  if tf_functions.ArgGroup.DIFFERENTIABLE not in func_info.arg_groups:
    return 

  operation, arg_values = value.operation_applications[0]
  arg_values_copy = [arg_value.copy() for arg_value in arg_values]

  # wrapps all arguments in variable
  for i, (arg_value_copy, arg_group) in enumerate(zip(arg_values_copy, func_info.arg_groups)):
    if arg_group is not tf_functions.ArgGroup.DIFFERENTIABLE: continue
    arg_value_copy.value = \
        tf.Variable(tf.cast(arg_value_copy.value, tf.float32), 
                    name=f'in{i}')

  with tf.GradientTape(persistent=True) as tape:
    output = operation.apply(
        arg_values_copy, settings_module.default_settings).value
    # squeeze output to a 1-D tensor;  
    squeezed_output = tf.reshape(output, [-1,])
    losses = [squeezed_output[i] for i in tf.range(squeezed_output.shape[0])]

  for arg_value, arg_value_copy, arg_name, arg_group, description \
      in zip(arg_values, 
             arg_values_copy, 
             dataflow_utils.separate_arg_names(operation),
             func_info.arg_groups, 
             func_info.description):

    if arg_group is not tf_functions.ArgGroup.DIFFERENTIABLE: continue

    provenance = [id_where(tape.gradient(
        loss, arg_value_copy.value)) for loss in losses]

    trace_info[arg_name]= \
        TraceSolution(provenance, description, False)


def parse_broadcast_ops(value, func_info, trace_info):
  """
  'tf.add(x, y)',
  'tf.divide(x, y)',
  'tf.equal(x, y)',
  'tf.greater(x, y)',
  'tf.greater_equal(x, y)',
  'tf.math.divide_no_nan(x, y)',
  'tf.maximum(x, y)',
  'tf.minimum(x, y)',
  'tf.multiply(x, y)',
  'tf.not_equal(x, y)',
  'tf.subtract(x, y)',
  are all treated as 'tf.add(x, y)'.
  """
  fake_value = value.copy()
  fake_value.operation_applications[0] = value_module.OperationApplication(
    all_operations.find_operation_with_name('tf.add(x, y)'),
    value.operation_applications[0].arg_values)
  fake_func_info = copy.deepcopy(func_info)
  fake_func_info.arg_groups[0] = tf_functions.ArgGroup.DIFFERENTIABLE
  fake_func_info.arg_groups[1] = tf_functions.ArgGroup.DIFFERENTIABLE
  parse_differentiable_args(fake_value, fake_func_info, trace_info)


def parse_along_axis_ops(value, func_info, trace_info):
  """
  'tf.argmax(input, axis)',
  'tf.argmin(input, axis)',
  'tf.reduce_max(input_tensor, axis)',
  'tf.reduce_mean(input_tensor, axis)',
  'tf.reduce_min(input_tensor, axis)',
  'tf.reduce_prod(input_tensor, axis)',
  'tf.reduce_sum(input_tensor, axis)',
  are all treated as 'tf.reduce_sum(input_tensor, axis)'.
  """
  operation, arg_values = value.operation_applications[0]
  fake_value = value.copy()
  fake_value.operation_applications[0] = value_module.OperationApplication(
    all_operations.find_operation_with_name('tf.reduce_sum(input_tensor, axis)'),
    arg_values)
  fake_func_info = copy.deepcopy(func_info)
  fake_func_info.arg_groups[0] = tf_functions.ArgGroup.DIFFERENTIABLE
  parse_differentiable_args(fake_value, fake_func_info, trace_info)
  # 
  if 'arg' in operation.name:
    trace_info['input'] = trace_info['input_tensor']
    trace_info.pop('input_tensor')


def parse_along_seg_ops(value, func_info, trace_info):
  """
  'tf.math.segment_max(data, segment_ids)',
  'tf.math.segment_mean(data, segment_ids)',
  'tf.math.segment_min(data, segment_ids)',
  'tf.math.segment_prod(data, segment_ids)',
  'tf.math.segment_sum(data, segment_ids)',
  are all treated as 'tf.math.segment_sum(data, segment_ids)'.
  """
  operation, arg_values = value.operation_applications[0]
  # treat 'data' (arg_values[0])
  fake_value = value.copy()
  if 'unsorted' in operation.name:
    new_operation = all_operations.find_operation_with_name(
          'tf.math.unsorted_segment_sum(data, segment_ids, num_segments)')
  else:
    new_operation = all_operations.find_operation_with_name(
          'tf.math.segment_sum(data, segment_ids)')
  fake_value.operation_applications[0] = \
      value_module.OperationApplication(new_operation, arg_values) # no side effect for arg_values
  fake_func_info = copy.deepcopy(func_info)
  fake_func_info.arg_groups[0] = tf_functions.ArgGroup.DIFFERENTIABLE
  parse_differentiable_args(fake_value, fake_func_info, trace_info)

  # treat 'segment_ids' (arg_values[1])
  arg_values = value.operation_applications[0].arg_values
  segment_ids = arg_values[1]
  if 'unsorted' in operation.name:
    max = arg_values[2].value
  else:
    max = int(segment_ids.max()) + 1
  mask = tf.equal(tf.range(max)[...,None], segment_ids.value[None,...])
  repeat_times = int(tf.reduce_prod(value.value[0].shape))
  repeated_mask = tf.stack([mask]*repeat_times, axis=1)
  provenance = tf.reshape(repeated_mask, shape=value.shape + segment_ids.shape)

  trace_info['segment_ids']= \
      TraceSolution(
          tensor_to_list(provenance, out_shape=value.value.shape), 
          func_info.description[1], False
      )


def parse_argsort_ops(value, func_info, trace_info):
  """
  'tf.argsort(values, axis, stable=True)'
  "tf.argsort(values, axis, direction='DESCENDING', stable=True)"
  """
  operation, arg_values = value.operation_applications[0]
  # treat 'data' (arg_values[0])
  fake_value = value.copy()
  if 'direction' in operation.name:
    new_operation = all_operations.find_operation_with_name(
        "tf.sort(values, axis, direction='DESCENDING')")
  else:
    new_operation = all_operations.find_operation_with_name(
        'tf.sort(values, axis)')
  fake_value.operation_applications[0] = \
      value_module.OperationApplication(new_operation, arg_values) # no side effect for arg_values
  fake_func_info = copy.deepcopy(func_info)
  fake_func_info.arg_groups[0] = tf_functions.ArgGroup.DIFFERENTIABLE
  parse_differentiable_args(fake_value, fake_func_info, trace_info)



def parse_gather_ops(value, func_info, trace_info):
  """
  'tf.gather(params, indices)',
  'tf.gather(params, indices, axis, batch_dims)',
  """
  # TODO: change indices from ArgGroup.VALUEWISE to ArgGroup.OTHERS 
  # to tackle 'tf.gather(params, indices, axis, batch_dims)'.
  output_zeros_like = tf.zeros_like(value.value)
  n_indices = output_zeros_like.shape[0]
  reshaped_zeros_like = tf.reshape(output_zeros_like, (n_indices, -1))
  # indices += tf.range()
  provenance = reshaped_zeros_like + tf.expand_dims(tf.range(n_indices), 1)
  squeezed_provenance = tf.reshape(provenance, (-1,1))
  trace_info['indices']= \
      TraceSolution(
          squeezed_provenance.numpy().tolist(), 
          func_info.description[1], False
      )


def parse_stacklike_ops(value, func_info, trace_info):
  """
  'tf.add_n(inputs)',
  'tf.stack(values, axis)',
  'tf.concat(values, axis)',
  """
  # treat 'data/input' (arg_values[0])
  operation, arg_values = value.operation_applications[0]

  # make sure no side effects for arg_values 
  arg_values_copy = [arg_value.copy() for arg_value in arg_values]
  
  arg0 = arg_values_copy[0].value

  # If arg0 is a sequence of sequence, unstack it into
  # a sequence of tensor. Otherwise, it is a sequence of 
  # tensor in the first place.
  if not arg_values_copy[0].elem_type_is_tensor:
    stacked_arg0 = tf.constant(arg0)
    arg0 = tf.unstack(stacked_arg0)

  # convert inner tensors of arg0 into variable for backprop
  arg0 = [tf.Variable(tf.cast(item, tf.float32),
       name=f'in{i}') for i, item in enumerate(arg0)]

  arg_values_copy[0].value = arg0

  with tf.GradientTape(persistent=True) as tape:
    output = operation.apply(
        arg_values_copy, settings_module.default_settings).value
    squeezed_output = tf.reshape(output, [-1,])
    losses = [squeezed_output[i] for i in tf.range(squeezed_output.shape)]

  provenance = [[id_where(tape.gradient(loss, variable)) 
      for variable in arg0] for loss in losses]

  dim_to_len = [0] + list(itertools.accumulate(
      int(tf.reduce_prod(tensor.shape)) for tensor in arg0))

  for i in range(len(provenance)):
    for j in range(len(provenance[i])):
      provenance[i][j] = list(map(lambda x: x + dim_to_len[j], provenance[i][j]))
    provenance[i] = list(itertools.chain(*provenance[i]))

  trace_info[dataflow_utils.separate_arg_names(operation)[0]]= \
      TraceSolution(provenance, func_info.description[0], False)


def parse_condition_ops(value, func_info, trace_info):
  """
  'tf.boolean_mask(tensor, mask)',
  'tf.where(condition)',
  """
  operation, arg_values = value.operation_applications[0]

  def get_condition_provenance(condition, repeat_times):
    condition_float = tf.Variable(tf.cast(condition, tf.float32))
    mask = tf.cast(condition_float, tf.bool)
    
    with tf.GradientTape(persistent=True) as tape:
      output = condition_float[mask]
      # squeeze output to a 1-D tensor;  
      squeezed_output = tf.reshape(output, [-1,])
      losses = [squeezed_output[i] 
          for i in tf.range(squeezed_output.shape[0]) for _ in range(repeat_times)]

    return [id_where(tape.gradient(loss, condition_float)) for loss in losses]


  if operation.name == 'tf.boolean_mask(tensor, mask)':
    condition = arg_values[1].value
    repeat_times = int(tf.reduce_prod(arg_values[0].value.shape[1:]))
    provenance = get_condition_provenance(condition, repeat_times)
    trace_info['mask']= \
        TraceSolution(provenance, func_info.description[1], False)   

  if operation.name == 'tf.where(condition)':
    condition = arg_values[0].value
    repeat_times = len(arg_values[0].value.shape)
    provenance = get_condition_provenance(condition, repeat_times)
    trace_info['condition']= \
        TraceSolution(provenance, func_info.description[0], False)   


def parse_others_ops(value, func_info, trace_info):
  """
  "tf.searchsorted(sorted_sequence, values, side='left')",
  "tf.searchsorted(sorted_sequence, values, side='right')"
  no provenance.
  """
  operation, arg_values = value.operation_applications[0]
  
  if operation.name == 'tf.math.bincount(arr)':
    pass
  elif operation.name == 'tf.math.count_nonzero(input)':
    pass
  elif operation.name == 'tf.math.count_nonzero(input, axis)':
    pass
  elif operation.name == 'tf.reduce_any(input_tensor, axis)':
    pass
  elif operation.name == 'tf.scatter_nd(indices, updates, shape)':
    pass
  elif operation.name == 'tf.tensor_scatter_nd_update(tensor, indices, updates)':
    pass
  elif operation.name == 'tf.unique_with_counts(x)':
    pass
  elif operation.name == 'tf.unstack(value, axis)':
    pass
  elif operation.name ==  'tf.gather_nd(params, indices)':
    pass
  elif operation.name == 'tf.gather_nd(params, indices, batch_dims)':
    pass


def trace(value: value_module.Value
    ) -> Dict[Tuple[Text, Callable, Text], TraceSolution]:

  operation, arg_values = value.operation_applications[0]

  # return None if the operation is not supported, including python
  # operations and some tensorflow operations that have yet to be 
  # implemented
  if operation.name not in tf_functions.TF_FUNCTIONS_INFO.keys():
    return {}

  assert operation.name in all_operations.get_operation_names(), \
      f'{operation.name} is not supported'

  trace_info = {}
  func_info = copy.deepcopy(tf_functions.TF_FUNCTIONS_INFO[operation.name])

  # parse VALUEWISE arguments
  parse_valuewise_args(value, func_info, trace_info)

  # parse all_EDGES arguments
  parse_all_edges_args(value, func_info, trace_info)

  # parse ELEMENTWISE arguments
  parse_elementwise_args(value, func_info, trace_info)

  # parse DIFFERENTIABLE arguments (i.e. output is differentiable w.r.t these arguments)
  parse_differentiable_args(value, func_info, trace_info)

  # parse BROADCAST operations
  if func_info.group is tf_functions.TraceGroup.BROADCAST:
    parse_broadcast_ops(value, func_info, trace_info)  

  # parse ALONG_AXIS operations
  if func_info.group is tf_functions.TraceGroup.ALONG_AXIS:
    parse_along_axis_ops(value, func_info, trace_info)
  
  # parse ALONG_SEG operations
  if func_info.group is tf_functions.TraceGroup.ALONG_SEG:
    parse_along_seg_ops(value, func_info, trace_info)

  # parse ARGSORT operations
  if func_info.group is tf_functions.TraceGroup.ARGSORT:
    parse_argsort_ops(value, func_info, trace_info)

  # parse GATHER operations
  if func_info.group is tf_functions.TraceGroup.GATHER:
    parse_gather_ops(value, func_info, trace_info)

  # parse STACKLIKE operations
  if func_info.group is tf_functions.TraceGroup.STACKLIKE:
    parse_stacklike_ops(value, func_info, trace_info)

  # parse STACKLIKE operations
  if func_info.group is tf_functions.TraceGroup.CONDITION:
    parse_condition_ops(value, func_info, trace_info)

  # parse OTHERS operations
  if func_info.group is tf_functions.TraceGroup.OTHERS:
    parse_others_ops(value, func_info, trace_info)

  # # assert all args are parsed
  assert len(trace_info) == operation.num_args, \
      f'len(trace_info) ({len(trace_info)}) != len(operation.num_args) ({operation.num_args})'

  return trace_info

      
# testing
if __name__ == '__main__':
  # value = value_module.ConstantValue(tf.constant([[3, 1, 2], [6, 5, 4]]))
  # axis = value_module.ConstantValue(1)
  # operation = all_operations.find_operation_with_name('tf.argsort(values, axis, stable=True)')
  # settings = settings_module.default_settings()
  # maybe_value = operation.apply([value, axis], settings)
  # print(trace(maybe_value))
  # print(maybe_value)
  
  # in1 = value_module.ConstantValue(tf.constant([2,1,1,1]))
  # in2 = value_module.ConstantValue(tf.constant([0,0,0,0]))
  # in3 = value_module.ConstantValue(5)
  # operation = all_operations.find_operation_with_name('tf.math.unsorted_segment_prod(data, segment_ids, '
  #   'num_segments)')
  # settings = settings_module.default_settings()
  # maybe_value = operation.apply([in1, in2, in3], settings)
  # print(trace(maybe_value))
  # print(maybe_value)

  in1 = value_module.ConstantValue(tf.constant([[2,1,1,1], [2,1,1,1], [2,1,1,1]]))
  in2 = value_module.ConstantValue(tf.constant([2,1,0]))
  operation = all_operations.find_operation_with_name('tf.gather(params, indices)')
  settings = settings_module.default_settings()
  maybe_value = operation.apply([in1, in2], settings)
  import pprint
  pprint.pprint(trace(maybe_value))
  pprint.pprint(maybe_value)
  