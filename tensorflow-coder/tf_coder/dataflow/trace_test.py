# Copyright 2021 The TF-Coder Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for value_search.py."""

from tf_coder.dataflow import trace

import tensorflow as tf

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from tf_coder.value_search import all_operations
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.dataflow import tf_functions


def trace_helper(func_name, args):
  argv = [value_module.InputValue(arg, name=name) for name, arg in args.items()]
  operation = all_operations.find_operation_with_name(func_name)
  value = operation.apply(argv, settings_module.default_settings())
  provenance = trace.trace(value)
  # discard description
  return {key:value.provenance for key, value in provenance.items()} 


def trace_helper_constant(func_name, args):
  argv = [value_module.ConstantValue(arg) for arg in args.values()]
  operation = all_operations.find_operation_with_name(func_name)
  value = operation.apply(argv, settings_module.default_settings())
  provenance = trace.trace(value)
  return set(repr(value.provenance) for value in provenance.values())


def get_unary_elementwise_operations():
  return list(key for key, value in tf_functions.TF_FUNCTIONS_INFO.items() 
      if value.arg_groups == [tf_functions.ArgGroup.ELEMENTWISE])

def get_broadcast_operations():
  return list(key for key, value in tf_functions.TF_FUNCTIONS_INFO.items() 
      if value.group == tf_functions.TraceGroup.BROADCAST)  

def get_along_axis_operations():
  return list(key for key, value in tf_functions.TF_FUNCTIONS_INFO.items() 
      if value.group == tf_functions.TraceGroup.ALONG_AXIS)

def get_along_seg_operations():
  return list(key for key, value in tf_functions.TF_FUNCTIONS_INFO.items() 
      if value.group == tf_functions.TraceGroup.ALONG_SEG and 'unsorted' not in key)

def get_along_unsorted_seg_operations():
  return list(key for key, value in tf_functions.TF_FUNCTIONS_INFO.items() 
      if value.group == tf_functions.TraceGroup.ALONG_SEG and 'unsorted' in key)


class TraceTest(parameterized.TestCase):

  def setUp(self):
    super(TraceTest, self).setUp()
    self.settings = settings_module.default_settings()

  # unary elementwise
  @parameterized.named_parameters(
      ('primitive',
          {'in1': 0.},
          {'in1': [[0]]}),

      ('vector', 
          {'in1': [1.,2.]},
          {'in1': [[0],[1]]}),

      ('matrix', 
          {'in1': [[1.,2.],[3.,4.]]},
          {'in1': [[0], [1], [2], [3]]}),

      ('tensor',
          {'in1': [[[1.,2.],[3.,4.]],[[1.,2.],[3.,4.]]]},
          {'in1': [[0], [1], [2], [3], [4], [5], [6], [7]]}),
  )
  def test_unary_elementwise(self, args, expected_provenance):
    operations = get_unary_elementwise_operations()
    for func_name in operations:
      provenance = trace_helper(func_name, args)
      self.assertEqual(provenance, expected_provenance)


  # BROADCASTE
  @parameterized.named_parameters(
      ('broadcast_3*4_3*1',
          {'in1': [[1.,1,1,1],[1,1,1,1],[1,1,1,1]],
           'in2': [[1.],[2],[4]]},
          {'in1': [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
           'in2': [[0], [0], [0], [0], [1], [1], [1], [1], [2], [2], [2], [2]]}),
  )
  def test_broadcast(self, args, expected_provenance):
    operations = get_broadcast_operations()
    for func_name in operations:
      provenance = trace_helper(func_name, args)
      self.assertEqual(provenance, expected_provenance)


  # ALONG_AXIS
  @parameterized.named_parameters(
      ('vector_along_axis0',
          {'input': [3,1,2], 
           'axis': 0},
          {'input': [[0, 1, 2]],
           'axis': None}),

      ('tensor_along_axis0', 
          {'input': [[[5,6,8],[3,1,5]],[[7,9,1],[5,2,1]]], 
           'axis': 0},
          {'input': [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
           'axis': None}),

      ('tensor_along_axis1', 
          {'input': [[[5,6,8],[3,1,5]],[[7,9,1],[5,2,1]]], 
           'axis': 1},
          {'input': [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]],
           'axis': None}),

      ('tensor_along_axis2',
          {'input': [[[5,6,8],[3,1,5]],[[7,9,1],[5,2,1]]], 
           'axis': 2},
          {'input': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
           'axis': None}),
  )
  def test_along_axis(self, args, expected_provenance):
    operations = get_along_axis_operations()
    for func_name in operations:
      provenance = trace_helper(func_name, args)
      self.assertEqual(provenance, expected_provenance)


  # ALONG_SEG (sorted)
  @parameterized.named_parameters(
      ('primitive_along_seg', 
          {'data': [1,2,3],
           'segment_ids': [0, 0, 2]},
          {'data': [[0, 1], [], [2]],
           'segment_ids': [[0, 1], [], [2]]}),

      ('vector_along_seg', 
          {'data': [[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]],
           'segment_ids': [0, 0, 2]},
          {'data': [[0, 4], [1, 5], [2, 6], [3, 7], [], [], [], [], [8], [9], [10], [11]],
           'segment_ids': [[0, 1], [0, 1], [0, 1], [0, 1], [], [], [], [], [2], [2], [2], [2]]}),

      ('matrix_along_seg',
          {'data': [[[1,2,3,4]], [[4, 3, 2, 1]], [[5,6,7,8]]],
           'segment_ids': [0, 0, 2]},
          {'data': [[0, 4], [1, 5], [2, 6], [3, 7], [], [], [], [], [8], [9], [10], [11]],
           'segment_ids': [[0, 1], [0, 1], [0, 1], [0, 1], [], [], [], [], [2], [2], [2], [2]]}),
  )
  def test_along_seg(self, args, expected_provenance):
    operations = get_along_seg_operations()
    for func_name in operations:
      provenance = trace_helper(func_name, args)
      self.assertEqual(provenance, expected_provenance)


  # ALONG_SEG (unsorted)
  @parameterized.named_parameters(
      ('primitive_along_seg', 
          {'data': [1,2,3],
           'segment_ids': [0, 1, 0],
           'num_segments': 2},
          {'data': [[0, 2], [1]],
           'segment_ids': [[0, 2], [1]],
           'num_segments': None}),

      ('vector_along_seg', 
          {'data': [[1,2,3,4], [5,6,7,8], [4,3,2,1]],
           'segment_ids': [0, 1, 0],
           'num_segments': 2},
          {'data': [[0, 8], [1, 9], [2, 10], [3, 11], [4], [5], [6], [7]],
           'segment_ids': [[0, 2], [0, 2], [0, 2], [0, 2], [1], [1], [1], [1]],
           'num_segments': None}),

      ('matrix_along_seg',
          {'data': [[[1,2,3,4]], [[5,6,7,8]], [[4,3,2,1]]],
           'segment_ids': [0, 1, 0],
           'num_segments': 2},
          {'data': [[0, 8], [1, 9], [2, 10], [3, 11], [4], [5], [6], [7]],
           'segment_ids': [[0, 2], [0, 2], [0, 2], [0, 2], [1], [1], [1], [1]],
           'num_segments': None}),

      ('vector_num_segments>#segment_ids',
          {'data': [2,1,1,1],
           'segment_ids': [0, 0, 0, 0],
           'num_segments': 5},
          {'data': [[0, 1, 2, 3], [], [], [], []],
           'segment_ids': [[0, 1, 2, 3], [], [], [], []],
           'num_segments': None}),
     
  )
  def test_along_seg(self, args, expected_provenance):
    operations = get_along_unsorted_seg_operations()
    for func_name in operations:
      provenance = trace_helper(func_name, args)
      self.assertEqual(provenance, expected_provenance)


  # STACKLIKE
  @parameterized.named_parameters(
      ('stack_primitive_axis0', 'tf.stack(values, axis)',
          {'input': [1,2],
           'axis': 0},
          {'input': [[0], [1]],
           'axis': None}),

      ('stack_vector_axis1', 'tf.stack(values, axis)',
          {'input': [[1,1],[2,2]],
           'axis': 1},
          {'input': [[0], [2], [1], [3]],
           'axis': None}),

      ('stack_matrix_axis1', 'tf.stack(values, axis)',
          {'input': [[[1],[2],[3]],[[1],[2],[3]]],
           'axis': 1},
          {'input': [[0], [3], [1], [4], [2], [5]],
           'axis': None}),

      ('concat_vector_axis0', 'tf.concat(values, axis)',
          {'input': [[1,1],[2,2]], 
           'axis': 0},
          {'input': [[0], [1], [2], [3]],
           'axis': None}),

      ('concat_matrix_axis1', 'tf.concat(values, axis)',
          {'input': [[[1],[2],[3]],[[1],[2],[3]]], 
           'axis': 1},
          {'input': [[0], [3], [1], [4], [2], [5]],
           'axis': None}),

      ('concat_seq_tensor_axis1', 'tf.concat(values, axis)',
          {'input': [tf.constant([[1, 2], [10, 20]]), tf.constant([[3, 4, 5], [30, 40, 50]])], 
           'axis': 1},
          {'input': [[0], [1], [4], [5], [6], [2], [3], [7], [8], [9]],
           'axis': None}),

      ('add_n_primitive', 'tf.add_n(inputs)',
          {'input': [1,3,1]},
          {'input': [[0, 1, 2]]}),

      ('add_n_vector', 'tf.add_n(inputs)',
          {'input': [[3, 5], [1, 6], [3, 5]]},
          {'input': [[0, 2, 4], [1, 3, 5]]}),

      ('add_n_matrix', 'tf.add_n(inputs)',
          {'input': [[[3, 5], [4, 8]], [[1, 6], [2, 9]], [[3, 5], [4, 8]]]},
          {'input': [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]}),
  )
  def test_stacklike(self, func_name, args, expected_provenance):
    provenance = trace_helper_constant(func_name, args)
    self.assertEqual(provenance, set(repr(value) for value in expected_provenance.values()))


  # CONDITION
  @parameterized.named_parameters(
      ('where_vector', 'tf.where(condition)',
          {'condition': [True, False, True]},
          {'condition': [[0], [2]]}),

      ('where_tensor', 'tf.where(condition)',
          {'condition': [[[True, False, True],[False, True, True]]]},
          {'condition': [[0], [0], [0], [2], [2], [2], [4], [4], [4], [5], [5], [5]]}),

      ('where_xy_vector', 'tf.where(condition, x, y)',
          {'condition': [True, False, True],
           'x': [1, 1, 1],
           'y': [2, 2, 2]},
          {'condition': [[0], [1], [2]],
           'x': [[0], [], [2]],
           'y': [[], [1], []]}),

      ('where_xy_tensor', 'tf.where(condition, x, y)',
          {'condition': [[[True, False, True],[False, True, True]]],
           'x': [[[1, 1, 1],[1, 1, 1]]],
           'y': [[[2, 2, 2],[2, 2, 2]]]},
          {'condition': [[0], [1], [2], [3], [4], [5]],
           'x': [[0], [], [2], [], [4], [5]],
           'y': [[], [1], [], [3], [], []]}),
  )
  def test_condition(self, func_name, args, expected_provenance):
    provenance = trace_helper(func_name, args)
    self.assertEqual(provenance, expected_provenance)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()
