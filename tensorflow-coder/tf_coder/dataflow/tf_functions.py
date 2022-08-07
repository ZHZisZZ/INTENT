import enum
import collections


# 'tf.math.segment_*(data, segment_ids)'
SEGMENT_DESCRIPTION = \
['One segment of data with the same id in `segment_ids`, from which {} are collected to the output tensor.',
 'The index in the output tensor to put the collected {} values']


# 'tf.math.unsorted_segment_*(data, segment_ids, num_segments)'
UNSORTED_SEGMENT_DESCRIPTION = SEGMENT_DESCRIPTION + ['']


def get_description(description_list, data, ph='&'):
  return ph.join(description_list).format(*data).split(ph)


@enum.unique
class ArgGroup(enum.Enum):
  # argument like axis or shape, applied on the value level instead of element level
  VALUEWISE = 'VALUEWISE'

  # argument like clip_value_min in tf.clip, and value in tf.fill 
  ALL_EDGES = 'ALL_EDGES'

  # output elements has one-to-one correspondence with this argument
  ELEMENTWISE = 'ELEMENTWISE'

  # output elements are differentiable w.r.t this argument
  DIFFERENTIABLE = 'DIFFERENTIABLE'

  OTHERS = 'OTHERS'


@enum.unique
class TraceGroup(enum.Enum):

  # all arguments are covered in NON_EDGES, ELEMENTWISE or DIFFERENTIABLE
  PREPROCESS_ONLY = 'PREPROCESS_ONLY'

  BROADCAST = 'BROADCAST'

  ALONG_AXIS = 'ALONG_AXIS'

  ALONG_SEG = 'ALONG_SEG'

  ARGSORT = 'ARGSORT'

  GATHER = 'GATHER'

  STACKLIKE = 'STACKLIKE'

  CONDITION = 'CONDITION'

  OTHERS = 'OTHERS'


FunctionInfo = collections.namedtuple(
    'FunctionInfo',
    ['group', 'description', 'arg_groups'])


TF_FUNCTIONS_INFO = {
  'tf.abs(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.add(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.add_n(inputs)': 
    FunctionInfo(group=TraceGroup.STACKLIKE,
                 description=['',],
                 arg_groups=[ArgGroup.OTHERS]),

  'tf.argmax(input, axis)': 
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.argmin(input, axis)':
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.argsort(values, axis, stable=True)': 
    FunctionInfo(group=TraceGroup.ARGSORT,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  "tf.argsort(values, axis, direction='DESCENDING', "
                        "stable=True)": 
    FunctionInfo(group=TraceGroup.ARGSORT,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.boolean_mask(tensor, mask)': 
    FunctionInfo(group=TraceGroup.CONDITION,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.OTHERS]),

  'tf.broadcast_to(input, shape)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.cast(x, dtype)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.ELEMENTWISE, ArgGroup.ALL_EDGES]),

  'tf.clip_by_value(t, clip_value_min, clip_value_max)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*3,
                 arg_groups=[ArgGroup.ELEMENTWISE, ArgGroup.ALL_EDGES, ArgGroup.ALL_EDGES]),

  'tf.concat(values, axis)': 
    FunctionInfo(group=TraceGroup.STACKLIKE,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.constant(value)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.divide(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.equal(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.exp(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.expand_dims(input, axis)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.eye(num_rows)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  'tf.eye(num_rows, num_columns)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.VALUEWISE]*2),

  'tf.eye(num_rows, dtype)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.VALUEWISE]*2),

  'tf.fill(dims, value)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.VALUEWISE, ArgGroup.ALL_EDGES]),

  'tf.gather(params, indices)': 
    FunctionInfo(group=TraceGroup.GATHER,
                 description=['the values gathered','index into `params` to gather values'],
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.OTHERS]),

  'tf.gather(params, indices, axis, batch_dims)': 
    FunctionInfo(group=TraceGroup.GATHER,
                 description=['',]*4,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE, ArgGroup.VALUEWISE, ArgGroup.VALUEWISE]),

  # auto backprop does not work
  # 'tf.gather_nd(params, indices)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  # 'tf.gather_nd(params, indices, batch_dims)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*3,
  #                arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE, ArgGroup.VALUEWISE]),

  'tf.greater(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.greater_equal(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  # 'tf.math.bincount(arr)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',],
  #                arg_groups=[ArgGroup.OTHERS]),

  'tf.math.ceil(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  # 'tf.math.count_nonzero(input)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',],
  #                arg_groups=[ArgGroup.OTHERS]),

  # 'tf.math.count_nonzero(input, axis)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.math.cumsum(x, axis)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.math.cumsum(x, axis, exclusive=True)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.math.divide_no_nan(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.math.floor(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.math.negative(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.math.reciprocal(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.math.reciprocal_no_nan(x)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.math.segment_max(data, segment_ids)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=get_description(SEGMENT_DESCRIPTION, ('max',)*2),
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),

  'tf.math.segment_mean(data, segment_ids)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=get_description(SEGMENT_DESCRIPTION, ('mean',)*2),
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),

  'tf.math.segment_min(data, segment_ids)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=get_description(SEGMENT_DESCRIPTION, ('min',)*2),
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),              

  'tf.math.segment_prod(data, segment_ids)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=get_description(SEGMENT_DESCRIPTION, ('production',)*2),
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),

  'tf.math.segment_sum(data, segment_ids)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=get_description(SEGMENT_DESCRIPTION, ('summation',)*2),
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),

  'tf.math.squared_difference(x, y)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.ELEMENTWISE]*2),

  # 'tf.math.top_k(input, k)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.math.unsorted_segment_max(data, segment_ids, '
    'num_segments)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=['',]*3,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.math.unsorted_segment_mean(data, segment_ids, '
   'num_segments)':
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=['',]*3,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.math.unsorted_segment_min(data, segment_ids, '
    'num_segments)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=['',]*3,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.math.unsorted_segment_prod(data, segment_ids, '
    'num_segments)': 
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=['',]*3,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.math.unsorted_segment_sum(data, segment_ids, '
    'num_segments)':
    FunctionInfo(group=TraceGroup.ALONG_SEG,
                 description=['',]*3,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.matmul(a, b)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE]*2),

  'tf.maximum(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.minimum(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.multiply(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.not_equal(x, y)': 
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  'tf.one_hot(indices, depth)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.VALUEWISE]*2),

  'tf.ones(shape)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  'tf.ones_like(input)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  "tf.pad(tensor, paddings, mode='CONSTANT')": 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  "tf.pad(tensor, paddings, mode='CONSTANT', "
                       "constant_values)": 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*3,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE, ArgGroup.VALUEWISE]),

  "tf.pad(tensor, paddings, mode='REFLECT')": 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  "tf.pad(tensor, paddings, mode='SYMMETRIC')": 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.range(start)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  'tf.range(start, limit, delta)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*3,
                 arg_groups=[ArgGroup.VALUEWISE]*3),

  # 'tf.reduce_any(input_tensor, axis)': 
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.reduce_max(input_tensor)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.DIFFERENTIABLE]),

  'tf.reduce_max(input_tensor, axis)': 
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]), # differentiable, but as if tf.reduce_mean was used

  'tf.reduce_mean(input_tensor)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.DIFFERENTIABLE]),

  'tf.reduce_mean(input_tensor, axis)': 
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.reduce_min(input_tensor)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.DIFFERENTIABLE]),

  'tf.reduce_min(input_tensor, axis)': 
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]), # differentiable, but as if tf.reduce_mean was used

  'tf.reduce_prod(input_tensor, axis)': 
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.reduce_sum(input_tensor)': 
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.DIFFERENTIABLE]),

  'tf.reduce_sum(input_tensor, axis)':
    FunctionInfo(group=TraceGroup.ALONG_AXIS,
                 description=['',]*2, 
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.reshape(tensor, shape)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2, 
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.reverse(tensor, axis)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2, 
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.roll(input, shift, axis)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*3, 
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE, ArgGroup.VALUEWISE]),

  'tf.round(x)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  # 'tf.scatter_nd(indices, updates, shape)':
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*3,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  # "tf.searchsorted(sorted_sequence, values, side='left')":
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),

  # "tf.searchsorted(sorted_sequence, values, side='right')":
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS]),

  'tf.sequence_mask(lengths)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  'tf.sequence_mask(lengths, maxlen)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.VALUEWISE, ArgGroup.VALUEWISE,]),

  'tf.shape(input)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  'tf.sign(x)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.sort(values, axis)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  "tf.sort(values, axis, direction='DESCENDING')":
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.sqrt(x)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.square(x)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.ELEMENTWISE]),

  'tf.squeeze(input)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.DIFFERENTIABLE]),

  'tf.squeeze(input, axis)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.stack(values, axis)':
    FunctionInfo(group=TraceGroup.STACKLIKE,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.subtract(x, y)':
    FunctionInfo(group=TraceGroup.BROADCAST,
                 description=['',]*2,
                 arg_groups=[ArgGroup.OTHERS]*2),

  # 'tf.tensor_scatter_nd_update(tensor, indices, updates)':
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*3,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.OTHERS, ArgGroup.DIFFERENTIABLE]),

  'tf.tensordot(a, b, axes)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*3,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.tile(input, multiples)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  'tf.transpose(a)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.DIFFERENTIABLE]),

  'tf.transpose(a, perm)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*2,
                 arg_groups=[ArgGroup.DIFFERENTIABLE, ArgGroup.VALUEWISE]),

  # 'tf.unique_with_counts(x)':
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',],
  #                arg_groups=[ArgGroup.OTHERS]),

  # 'tf.unstack(value, axis)':
  #   FunctionInfo(group=TraceGroup.OTHERS,
  #                description=['',]*2,
  #                arg_groups=[ArgGroup.OTHERS, ArgGroup.VALUEWISE]),

  'tf.where(condition)':
    FunctionInfo(group=TraceGroup.CONDITION,
                 description=['',],
                 arg_groups=[ArgGroup.OTHERS]),

  'tf.where(condition, x, y)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',]*3,
                 arg_groups=[ArgGroup.ELEMENTWISE, ArgGroup.DIFFERENTIABLE, ArgGroup.DIFFERENTIABLE]),

  'tf.zeros(shape)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),

  'tf.zeros_like(input)':
    FunctionInfo(group=TraceGroup.PREPROCESS_ONLY,
                 description=['',],
                 arg_groups=[ArgGroup.VALUEWISE]),
}


if __name__ == '__main__':
  print(get_description(SEGMENT_DESCRIPTION, ('max',)*2))