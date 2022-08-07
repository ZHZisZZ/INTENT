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
"""A script for using TF-Coder (an alternative to using the Colab notebook).

Usage:
1. Edit `get_problem()` to specify your problem.
2. If desired, edit `get_settings()` to specify settings for TF-Coder.
3. Run this file, e.g., `python3 tf_coder_main.py`.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU is faster than GPU.

from absl import app  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf  # pylint: disable=unused-import

from tf_coder.dataflow import dataflow
from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder import asyn_utils


# def get_problem():
#   """Specifies a problem to run TF-Coder on. Edit this function!"""
#   # Input tensor lists
#   #! key should match for each input tensor 
#   inputs_list = [
#     [
#     [[-1.37, 2.37, -4.07, -3.63, 3.29],
#       [-1.32, 1.19, 3.87, 3.14, 3.44],
#       [2.61, 2.64, 0.95, 1.51, 0.1],
#       [-1.03, 0.91, -0.32, 2.95, 1.26]]
#     ]
#   ]

#   # Output tensor lists
#   output_list = [
#     [[0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0],
#      [0, 0, 1, 0, 1],
#      [0, 1, 0, 0, 0]]
#   ]

#   # A list of relevant scalar constants (if any).
#   constants = []

#   # An English description of the tensor manipulation.
#   description = 'identify elements between 0 and 1'

#   return inputs_list, output_list, constants, description

def get_problem():
  """Specifies a problem to run TF-Coder on. Edit this function!"""
  # Input tensor lists
  #! key should match for each input tensor 
  inputs_list = [
    [[1,1,1,1]],
    [[3,1,2,1]],
    # [[10,2,1,3]]
  ]

  # Output tensor lists
  output_list = [
    [0,0,0,0],
    [3,0,2,0],
    # [10,2,0,3],
  ]

  # A list of relevant scalar constants (if any).
  constants = []

  # An English description of the tensor manipulation.
  description = 'set every 1 to 0'

  return inputs_list, output_list, constants, description


def get_settings(
    time_limit=300,
    number_of_solutions=3,
    solution_requirement='all inputs'):
  """Specifies settings for TF-Coder. Edit this function!"""
  assert solution_requirement in ['all inputs', 'one input', 'no restriction']

  return settings_module.from_dict({
      'timeout': time_limit,
      'only_minimal_solutions': False,
      'max_solutions': number_of_solutions,
      'require_all_inputs_used': solution_requirement == 'all inputs',
      'require_one_input_used': solution_requirement == 'one input',
  })


def run_tf_coder(inputs_list, output_list, constants, 
                 description, settings, asyn=None):
  """Runs TF-Coder on a problem, using the given settings."""
  # Results will be printed to standard output.
  return colab_interface.run_value_search_from_colab(
      inputs_list, output_list, constants, 
      description, settings, asyn)


def print_supported_operations():
  """Run this function to print all supported operations."""
  colab_interface.print_supported_operations()


def main(unused_argv):
  import json
  colab_interface.warm_up()

  inputs_list, output_list, constants, description = get_problem()
  settings = get_settings()
  
  asyn = asyn_utils.Asyn()
  asyn.aborted = True
  # asyn = None
  result = run_tf_coder(inputs_list, output_list, constants, 
                        description, settings, asyn)

  nodes, edges, trace_edges = \
      dataflow.dataflow_generator(result.solutions[1].value)


if __name__ == '__main__':
  app.run(main)
