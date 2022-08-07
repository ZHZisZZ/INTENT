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

from tf_coder.dataflow.dataflow import dataflow_generator

import tensorflow as tf

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from tf_coder.value_search import all_operations
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search_settings as settings_module


class DataFlowTest(parameterized.TestCase):

  def setUp(self):
    super(DataFlowTest, self).setUp()
    self.add_operation = all_operations.find_operation_with_name('tf.add(x, y)')
    self.settings = settings_module.default_settings()
    self.in1 = value_module.InputValue(tf.constant([[1,2,3]]), name='in1')
    self.in2 = value_module.InputValue(tf.constant([[1],[2],[3]]), name='in2')

  def _check_no_type_printed(self, nodes):
    for key, value in nodes.items():
      self.assertFalse('tf' in value['value'] if 'n' in key else False)

  def test_same_input_used_multiple_times(self):
    result = self.add_operation.apply([self.in1, self.in1], self.settings)
    nodes, edges, trace_edges = dataflow_generator(result)
    self.assertLen(nodes, 4)
    self.assertLen(edges, 3)
    self._check_no_type_printed(nodes)

  def test_different_inputs_with_same_value(self):
    result = self.add_operation.apply([self.in1, self.in2], self.settings)
    nodes, edges, trace_edges = dataflow_generator(result)
    self.assertLen(nodes, 4)
    self.assertLen(edges, 3)  
    self._check_no_type_printed(nodes)

  def test_same_intermediate_values_used_multiple_times(self):
    _1p2 = self.add_operation.apply([self.in1, self.in2], self.settings)
    _1p2p1p2 = self.add_operation.apply([_1p2, _1p2], self.settings)
    nodes, edges, trace_edges = dataflow_generator(_1p2p1p2)
    self.assertLen(nodes, 8)
    self.assertLen(edges, 9)
    self._check_no_type_printed(nodes)
    print()
  
  def test_pair_operation(self):
    operation = all_operations.find_operation_with_name('PairCreationOperation')
    result = operation.apply([self.in1, self.in2], self.settings)
    nodes, edges, trace_edges = dataflow_generator(result)
    self._check_no_type_printed(nodes)
  

if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()
  
  # add_operation = all_operations.find_operation_with_name('tf.add(x, y)')
  # settings = settings_module.default_settings()
  # in1 = value_module.InputValue(tf.constant([[1,2,3]]), name='in1')
  # in2 = value_module.InputValue(tf.constant([[1],[2],[3]]), name='in2')
  # _1p2 = add_operation.apply([in1, in2], settings)
  # _1p2p1p2 = add_operation.apply([_1p2, _1p2], settings)
  # nodes, edges, trace_edges = dataflow_generator(_1p2p1p2)
  # import pprint
  # pprint.pprint(nodes)
  # pprint.pprint(edges)
  # pprint.pprint(trace_edges)
