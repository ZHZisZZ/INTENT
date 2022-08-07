from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf
import ast
from typing import Tuple, List, Text, Type, Dict
from tf_coder.value_search import value as value_module
from tf_coder.value_search import operation_base as operation_module
from tf_coder.value_search import all_operations as all_operations_module
from tf_coder.value_search import function_operation as function_operation_module
from tf_coder.value_search import python_operations as python_operations_module


def get_docstring(operation: operation_module.Operation):
  """Return the docstring for each operations."""
  docstring = operation.metadata.docstring

  if isinstance(operation, function_operation_module.FunctionOperation):
    docstring = docstring[:docstring.find('\n')]
  else:
    docstring = docstring[1:docstring.find('\n',2)]

  return docstring


def separate_arg_names(operation: operation_module.Operation) -> List[Text]:
  """Separate argument names from operation docstring."""
  operation_name = operation.name
  bp = operation_name.find('(')
  ep = operation_name.find(')')
  if bp != -1 and ep != -1:
    arg_string = operation_name[bp+1:ep]
    return arg_string.split(', ')
  return [''] * operation.num_args


def value_hash(value: value_module.Value) -> Tuple[Text, Type, Text]:
  """Return the hashable representation of 'value'."""
  name = value.name \
      if isinstance(value, value_module.InputValue) else None
  return (repr(value), value.__class__, name)


def tensor_to_string(tensor, decimals=2):
  """Converts a tensor into a string representation used for equality tests.

  TF-Coder considers two tensors to be equal if and only if their string
  representations (as computed by this function) are equal.

  Args:
    tensor: A Tensor.
    decimals: The number of floating-point decimal places to consider.

  Returns:
    A string representation of the tensor.
  """
  np_array = tensor.numpy()
  if np_array.dtype in [np.float32, np.float64]:
    # casting as np.float64 allows tolist() to keep the decimals in numpy
    np_array = np.around(np_array.astype(np.float64), decimals=decimals)

  # str(np_array.tolist()) is significantly faster than str(np_array).
  return str(np_array.tolist())


def object_to_string(obj, decimals=2):
  """Converts an object into a string representation used for equality tests.

  TF-Coder considers two objects to be equal if and only if their string
  representations (as computed by this function) are equal.

  Note that two sequences are considered the same if their elements are the
  same, even if one sequence is a list and the other is a tuple.

  Args:
    obj: An object, which could be a Tensor, SparseTensor, TensorFlow dtype,
      primitive (int, float, bool, or string), or sequence (list, tuple, or
      namedtuple) of other such objects.
    decimals: As described in tensor_to_string().

  Returns:
    A string representation of the object.

  Raises:
    ValueError: If `obj` has an unsupported type.
  """
  # Tensors.
  if isinstance(obj, tf.Tensor):
    return tensor_to_string(obj)
  if isinstance(obj, tf.SparseTensor):
    # TODO(kshi): Round float SparseTensors according to `decimals`.
    return str(obj)

  obj_type = type(obj)

  # Primitives and TensorFlow dtypes are handled the same way (with repr()).
  if obj_type in (int, float, bool, str, tf.DType):
    if obj_type == float:
      # Floats must be rounded.
      obj = round(obj, decimals)
    return repr(obj)

  # Sequences (lists, tuples, and namedtuples) of supported objects.
  if obj_type == list or isinstance(obj, tuple):
    return '(' + ', '.join(object_to_string(elem, decimals=decimals)
                              for elem in obj) + ')'

  # All other types are currently unsupported.
  raise ValueError('object_to_string called with unsupported object; type={} '
                   'and str={}.'.format(obj_type, obj))
  
  
################################################################################
ParsedStruct = namedtuple('ParsedStruct', ['operation_name', 'list_of_args', 'dict_of_consts'])
StoredStruct = namedtuple('StoredStruct', ['operation', 'parsed_struct'])

operation_name_to_operation : Dict[Text, List[StoredStruct]] = defaultdict(list)


def _split_operation_name_and_args(expression, sign=','):
  """Assume open, end and sign exist in expression."""
  # rp = 0
  stack = []
  list_of_indices = []
  open2close = {'(':')', '[':']'}
  close2open = {')':'(', ']':'['}
  # open = None
  end = None
  
  for i, c in enumerate(expression):
    if c == sign and len(stack) in [0, 1]:
      list_of_indices.append(i)
    elif c in ['(', '[']:
      # open = c if not open else open
      stack.append(c)
    elif c in [')', ']']:
      end = c
      top = stack.pop()
      if top != close2open[c]:
        raise ValueError('Bracket/Parethesis does not match.')
  
  # open_index = expression.index(open)
  # close_index = expression.rindex(open2close[open])
  open_index = expression.index(close2open[end])
  close_index = expression.rindex(end)
    
  
  list_of_indices.insert(0, open_index)
  
  operation_name = expression[:open_index].strip()
    
  args_by_comma = [expression[start+1: end].strip()
                    for start, end 
                    in zip(list_of_indices, 
                           list_of_indices[1:] + [close_index])]
  
  return operation_name, end, args_by_comma
  

def _parse_python_indexingslicing_expression(expression: Text):
  supported_indexing = \
  """
  arg0[arg1]
  arg0[:, arg1]
  """
  supported_slicing = \
  """
  arg0[arg1:]
  arg0[:arg1]
  arg0[arg1:arg2]
  arg0[:, arg1:]
  arg0[:, :arg1]
  arg0[:, arg1:arg2]
  """
  arg0, _, raw_args_in_bracket = _split_operation_name_and_args(expression)
  
  operation_name = None
  list_of_args = [arg0]
  dict_of_consts = {}

  indexing = False
  skip_first_dimension = False
  raw_args = raw_args_in_bracket[0]
  
  if len(raw_args_in_bracket) == 2:
    skip_first_dimension = True
    raw_args = raw_args_in_bracket[1]
  
  # construct a fake open ( and a close ) to look like a function call
  raw_args = f'({raw_args})'
  
  _, _, args_by_colon = _split_operation_name_and_args(raw_args, ':')
    
  # there is no colon -> indexing operation
  if len(args_by_colon) == 1:
    indexing = True
    list_of_args.append(args_by_colon[0])
    if not skip_first_dimension:
      operation_name = python_operations_module.IndexingOperation
    else:
      operation_name = python_operations_module.IndexingAxis1Operation
      
  else:      
    # slicing operation
    list_of_args.extend(filter(lambda x:x, args_by_colon))
    if not skip_first_dimension:
      if args_by_colon[0] and args_by_colon[1]:
        operation_name = python_operations_module.SlicingAxis0BothOperation
      elif args_by_colon[0]:
        operation_name = python_operations_module.SlicingAxis0LeftOperation
      elif args_by_colon[1]:
        operation_name = python_operations_module.SlicingAxis0RightOperation
    else:
      if args_by_colon[0] and args_by_colon[1]:
        operation_name = python_operations_module.SlicingAxis1BothOperation
      elif args_by_colon[0]:
        operation_name = python_operations_module.SlicingAxis1LeftOperation
      elif args_by_colon[1]:
        operation_name = python_operations_module.SlicingAxis1RightOperation

  if not operation_name:
    if indexing:
      raise NotImplementedError(
          "{} indexing is not supported.\n"
          "Supported indexing: \n{}".format(expression, supported_indexing))
    else:
      raise NotImplementedError(
          "{} slicing is not supported.\n"
          "Supported slicing: \n{}".format(expression, supported_slicing))
      
  return ParsedStruct(operation_name, list_of_args, dict_of_consts)
    

def _parse_python_tuple_creation_expression(expression: Text):
  supported_tuple_creation = \
  """
  (arg0,)
  (arg0, arg1)
  (arg0, arg1, arg2)
  """
  _, _, args = _split_operation_name_and_args(expression)
  args = list(filter(lambda arg: arg, args))
  if len(args) == 1:
    return ParsedStruct(python_operations_module.SingletonTupleCreationOperation, args, {})
  elif len(args) == 2:
    return ParsedStruct(python_operations_module.PairCreationOperation, args, {})
  elif len(args) == 3:
    return ParsedStruct(python_operations_module.TripleCreationOperation, args, {})
  else:
    raise NotImplementedError(
        "{} tuple creation is not supported.\n"
        "Supported slicing: \n{}".format(expression, supported_tuple_creation))
  
  
def parse_expression(expression: Text):
  """Takes an expression and return ParsedStructure

  Args:
    expression: an expression. For example, 'tf.cast(tf.transpose(tf.cast(in1, tf.int64)), tf.int32)'

  Returns:
    A tuple (function_name, list_of_args, dict_of_consts), where function_name
    is a string, list_of_args is a list of strings, and list_of_consts is a
    list of constants. For example, if the expression 'tf.foo.bar(x, axis, baz=True)', 
    then this function would return ('tf.foo.bar', ['x', 'axis'], {'baz': True}).

  Raises:
    ValueError: If the FunctionInfo's name is not properly formatted.
  """
  operation_name, list_of_args, dict_of_consts = None, [], {}
  
  if expression.count('(') != expression.count(')') and \
     expression.count('[') != expression.count(']'):
    raise ValueError("The expression must have exactly the same number of "
                     "opening parenthesis/bracket and closing parenthesis/bracket.")
  
  # constant or input name
  if not expression.count('(') and not expression.count('['):
    return  ParsedStruct(None, [], [])
    
  operation_name, end, args_by_comma = _split_operation_name_and_args(expression)
  
  # begins with ( -> tuple creation operation
  
  if not operation_name:
    return _parse_python_tuple_creation_expression(expression)
  
  # token followed by [ -> indexing/slicing operation
  elif end == ']':
    return _parse_python_indexingslicing_expression(expression)
  
  
  for split in args_by_comma:
    if '=' in split and \
       ('(' not in split or split.index('=') < split.index('(')) and \
       ('[' not in split or split.index('=') < split.index('[')):
      equal = split.index('=')
      kwarg_name = split[:equal].strip()
      literal_as_string = split[equal+1:].strip()
      # const or kwarg (can not eval())
      dict_of_consts[kwarg_name] = literal_as_string
    else:
      list_of_args.append(split)
      
  return ParsedStruct(operation_name, list_of_args, dict_of_consts)
  
  
def find_operation_with_parsed_struct(parsed_struct: ParsedStruct):
  if not operation_name_to_operation:
    # TODO: Python function (slicing, tuple/triple) not supported
    operation_list = all_operations_module.get_tf_operations()
    for operation in operation_list:
      function_name = operation.function_name
      arg_names = operation.arg_names
      constant_kwargs = operation.constant_kwargs
      operation_name_to_operation[function_name].append(
          StoredStruct(operation, 
                       ParsedStruct(function_name, 
                                    arg_names, 
                                    constant_kwargs)))
      
  # python operation 
  if not isinstance(parsed_struct.operation_name, str):
    return parsed_struct.operation_name()

  if parsed_struct.operation_name not in operation_name_to_operation:
    raise ValueError(f'{parsed_struct.operation_name} not supported.')
  
  
  for candidate_operation, canditate_parsed_struct in \
      operation_name_to_operation[parsed_struct.operation_name]:
    
    # number of pargs of the target expression should 
    # be less than that of the candidate operation
    if len(parsed_struct.list_of_args) > len(canditate_parsed_struct.list_of_args):
      continue
    
    # pargs of the candidate operation should be found in 
    # the pargs or kwargs of the target expression
    set1 = set(canditate_parsed_struct.list_of_args[len(parsed_struct.list_of_args):])
    set2 = set(parsed_struct.dict_of_consts.keys())
    if not set1.issubset(set2):
      continue
    
    # kwargs of target expression should be either correct positional
    # args or correct const kwargs of the candidate operation 
    set1 = set(parsed_struct.dict_of_consts.keys())
    set2 = set().union(canditate_parsed_struct.list_of_args, 
                       canditate_parsed_struct.dict_of_consts.keys())
    if not set1.issubset(set2):
      continue
    
    OK = True
    for c_kwarg_name, c_kwarg_val in canditate_parsed_struct.dict_of_consts.items():
      if c_kwarg_name in parsed_struct.dict_of_consts and \
          ast.literal_eval(
            parsed_struct.dict_of_consts[c_kwarg_name]) != c_kwarg_val:
        OK = False; break
    if not OK: continue
      
    parsed_struct.dict_of_consts.update(canditate_parsed_struct.dict_of_consts)

    for i in range(len(parsed_struct.list_of_args), 
                   len(canditate_parsed_struct.list_of_args)):
      literal_as_string = parsed_struct.dict_of_consts.pop(
          canditate_parsed_struct.list_of_args[i])
      parsed_struct.list_of_args.append(literal_as_string)
      
    return candidate_operation

  potential_stored_structs = operation_name_to_operation[parsed_struct.operation_name]
  potential_operations = [struct.operation.name for struct in potential_stored_structs]

  raise ValueError("Operation not supported. \n"
                   "Supported usage: \n{}".format('\n'.join(potential_operations)))
  


if __name__ == "__main__":  
  indexing_slicing_expressions = [
    'arg0[arg1]',
    'arg0[:, arg1]',
    'arg0[arg1:]',
    'arg0[:arg1]',
    'arg0[arg1:arg2]',
    'arg0[:, arg1:]',
    'arg0[:, :arg1]',
    'arg0[:, arg1:arg2]',
  ]
  for e in indexing_slicing_expressions:
    print(e)
    parsed_struct = parse_expression(e)
    print(parsed_struct)
    print(find_operation_with_parsed_struct(parsed_struct))
    
  tuple_creation_expressions = [
    '(arg0,)',
    '(arg0, arg1)',
    '(arg0, arg1, arg2)',
  ]
  for e in tuple_creation_expressions:
    print(e)
    print(_split_operation_name_and_args(e))
  
  # e = 'tf.argmax(in1, axis=1)'
  # parsed_struct = parse_expression(e)
  # print(find_operation_with_parsed_struct(parsed_struct))
