import tensorflow as tf

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from tf_coder.dataflow import dataflow_utils
from tf_coder.value_search import python_operations


class DataflowUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(DataflowUtilsTest, self).setUp()
    
  @parameterized.named_parameters(
      ('simple', 
          'tf.func(in1)', 
          dataflow_utils.ParsedStruct(
            'tf.func', 
            ['in1'],
            {}
          )
      ),
      
      ('simple_with_multiple_spaces', 
          'tf.func( in1 )   ', 
          dataflow_utils.ParsedStruct(
            'tf.func', 
            ['in1'],
            {}
          )
      ),
      
      ('multiple_args_with_kwargs', 
          'tf.func(in1, in2, in3, in4=in4)   ', 
          dataflow_utils.ParsedStruct(
            'tf.func', 
            ['in1', 'in2', 'in3'],
            {'in4': 'in4'}
          )
      ),
      
      ('composition_1_with_spaces', 
          'tf.func(in1, in2, tf.func2(in3, in5) , in4 = in4)   ', 
          dataflow_utils.ParsedStruct(
            'tf.func', 
            ['in1', 'in2', 'tf.func2(in3, in5)'],
            {'in4': 'in4'}
          )
      ),
      
      ('composition_2', 
          'tf.func(in1, in2, tf.func2(in3, in5=in5), in4=in4)', 
          dataflow_utils.ParsedStruct(
            'tf.func', 
            ['in1', 'in2', 'tf.func2(in3, in5=in5)'],
            {'in4': 'in4'}
          )
      ),
      
      ('composition_3', 
          'tf.func(in1, in2, in3=tf.func2(in3, in5=in5), in4=in4)', 
          dataflow_utils.ParsedStruct(
            'tf.func', 
            ['in1', 'in2'],
            {'in3': 'tf.func2(in3, in5=in5)', 'in4': 'in4'}
          )
      ),
      
      ('tuple_simple', 
          '(in1, in2)', 
          dataflow_utils.ParsedStruct(
            python_operations.PairCreationOperation, 
            ['in1', 'in2'],
            {}
          )
      ),
      
      ('tuple_composition_1', 
          '(in1, (in2, in3), in4)', 
          dataflow_utils.ParsedStruct(
            python_operations.TripleCreationOperation, 
            ['in1', '(in2, in3)', 'in4'],
            {}
          )
      ),
      
      ('tuple_composition_2', 
          '(tf.constant(in1, (in1, in1)), (in2, in3), in4)', 
          dataflow_utils.ParsedStruct(
            python_operations.TripleCreationOperation, 
            ['tf.constant(in1, (in1, in1))', '(in2, in3)', 'in4'],
            {}
          )
      ),
      
      ('tuple_composition_3', 
          '((tf.constant(in1, (in1, in1)), (in2, in3), in4), )', 
          dataflow_utils.ParsedStruct(
            python_operations.SingletonTupleCreationOperation, 
            ['(tf.constant(in1, (in1, in1)), (in2, in3), in4)'],
            {}
          )
      ),
      
      ('tuple_composition_4', 
          'tf.foo((in2, in3), in4, a=tf.constant(in1, (in1, in1)))', 
          dataflow_utils.ParsedStruct(
            'tf.foo',
            ['(in2, in3)', 'in4'],
            {'a':'tf.constant(in1, (in1, in1))'}
          )
      ),
      
      ('indexing_simple', 
          'in1[:1]', 
          dataflow_utils.ParsedStruct(
            python_operations.SlicingAxis0RightOperation,
            ['in1' , '1'],
            {}
          )
      ),
      
      ('indexing_composition_1', 
          'in1[:, :tf.max(in1, in2)]', 
          dataflow_utils.ParsedStruct(
            python_operations.SlicingAxis1RightOperation,
            ['in1', 'tf.max(in1, in2)'],
            {}
          )
      ),
      
      ('indexing_composition_2', 
          'in1[:, tf.min(in1, in2) : tf.max(in1, in2)]', 
          dataflow_utils.ParsedStruct(
            python_operations.SlicingAxis1BothOperation,
            ['in1', 'tf.min(in1, in2)', 'tf.max(in1, in2)'],
            {}
          )
      ),
      
      ('indexing_composition_3', 
          'in1[:, in1[: tf.min(in1, in2)] : tf.max(in1, in2)]', 
          dataflow_utils.ParsedStruct(
            python_operations.SlicingAxis1BothOperation,
            ['in1', 'in1[: tf.min(in1, in2)]', 'tf.max(in1, in2)'],
            {}
          )
      ),
      
      ('indexing_composition_4',
          'tf.constant(a=in1[:, in1[: tf.min(in1, in2)] : tf.max(in1, in2)])', 
          dataflow_utils.ParsedStruct(
            'tf.constant',
            [],
            {'a':'in1[:, in1[: tf.min(in1, in2)] : tf.max(in1, in2)]'}
          )
      ),
      
      ('indexing_after_tf',
          'tf.shape(in1)[1]',
          dataflow_utils.ParsedStruct(
            python_operations.IndexingOperation,
            ['tf.shape(in1)', '1'],
            {}
          )
      ),      
  )
  def test_parse_expression(
      self, expression, target_parsed_struct):
    parsed_struct = \
        dataflow_utils.parse_expression(expression)
    self.assertEqual(parsed_struct, target_parsed_struct)
    
    
  @parameterized.named_parameters(
      ('simple', 'tf.abs(in1)', 'tf.abs(x)'),
      
      ('simple_with_kwarg', 'tf.abs(x=in1)', 'tf.abs(x)'),
      
      ('kwarg_with_constant_kwarg', 
          'tf.argsort(in1, axis=1, stable=True)', 
          'tf.argsort(values, axis, stable=True)'),
      
      ('missing_constant_kwarg', 
          'tf.argsort(in1, 1)',
          'tf.argsort(values, axis, stable=True)'),
      
      ('kwarg_with_missing_constant_kwarg', 
          'tf.argsort(values=in1, axis=1)',
          'tf.argsort(values, axis, stable=True)'),
      
      ('constant_kwarg_with_different_values', 
          "tf.searchsorted(in1, in2, side='left')",
          "tf.searchsorted(sorted_sequence, values, side='left')"),
      
      ('all_kwarg_with_swap', 
          "tf.searchsorted(values=in2, sorted_sequence=in1, side='left')",
          "tf.searchsorted(sorted_sequence, values, side='left')"),
  )
  def test_find_operation_with_parsed_struct(
      self, expression, operation_name):
    parsed_struct = \
        dataflow_utils.parse_expression(expression)
    operation = dataflow_utils.find_operation_with_parsed_struct(parsed_struct)  
    self.assertEqual(operation.name, operation_name)
    

if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()