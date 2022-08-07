import numpy as np
from typing import Text, Dict
from tf_coder.value_search import value as value_module


def dataflow_dict(
    value: value_module.Value,
    parent_expression: Text = 'null', 
) -> Dict[Text, Text]:
  """Return the dataflow structure of a synthesized expression.

  Args:
    value: The output value of tf-coder, a recursive structure that contains 
        all the intermediate values that we need to create the dataflow structure.

    parent_name: The expression of the parent node.


  Key: 
    "expression": The expression (accumulated operations) at the node.

    "value": The intermediate value produced by the node expression. 
      Constant values have no "value" key since their "name" are 
      equal to their "value".

    "operation": The operation applied at the node. 
      Constant values and input values have no "operation" keys.

    "parent": The expression of the parent node. 
      "null" if the node is the root.

  =====================================================================
  For example, 
  input = [
    'row': [10, 20, 30]
    'col': [5, 6, 7, 8]
  ]
  output = 
    [[15, 16, 17, 18],
     [25, 26, 27, 28],
     [35, 36, 37, 38]]
  synthesized_program = "tf.add(cols, tf.expand_dims(rows, 1))"

  Then, the dataflow structure:
  {
    "expression": "tf.add(cols, tf.expand_dims(rows, 1))",
    "value": "tf.int32:[[15, 16, 17, 18], [25, 26, 27, 28], [35, 36, 37, 38]]",
    "operation": "tf.add(x, y)",
    "parent": "null",
    "children": [
      {
        "expression": "cols",
        "value": "tf.int32:[5, 6, 7, 8]",
        "parent": "tf.add(cols, tf.expand_dims(rows, 1))"
      },
      {
        "expression": "tf.expand_dims(rows, 1)",
        "value": "tf.int32:[[10], [20], [30]]",
        "operation": "tf.expand_dims(input, axis)",
        "parent": "tf.add(cols, tf.expand_dims(rows, 1))",
        "children": [
          {
            "expression": "rows",
            "value": "tf.int32:[10, 20, 30]",
            "parent": "tf.expand_dims(rows, 1)"
          },
          {
            "expression": "1",
            "parent": "tf.expand_dims(rows, 1)"
          }
        ]
      }
    ]
  }
  """
  assert isinstance(value, value_module.Value)
  expression = value.reconstruct_expression()

  if isinstance(value, value_module.ConstantValue):
    return {'expression': expression, 'parent': parent_expression}

  if isinstance(value, value_module.InputValue):
    return {'expression': expression, 'value': str(value), 'parent': parent_expression}

  operation = value.operation_applications[0].operation.name
  node = {'expression': expression, 
          'value': str(value),
          'operation': operation,
          'parent': parent_expression, 
          'children': []}
  for arg_value in value.operation_applications[0].arg_values:
    node['children'].append(dataflow_dict(arg_value, expression))
  return node