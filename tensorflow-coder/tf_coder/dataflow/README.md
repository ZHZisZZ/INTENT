## Documentation for `dataflow_generator()` function in tf_coder/dataflow/dataflow.py.

`dataflow_generator()` are tested successfully on most `google_benchmark`.

For example, the return value of `dataflow_generator()` for `google_06` benchmark is
```
tf.math.segment_max(in2, in1)
{
  "nodes": {
    "n0": {
      "type": "intermediate",
      "value": "[1, 4, 5, 10]",
      "label": "tf.math.segment_max(in2, in1)"
    },
    "n1": {
      "type": "input",
      "value": "[1, 3, 4, 5, 10, 8, 9, 4]",
      "label": "in2"
    },
    "n2": {
      "type": "input",
      "value": "[0, 1, 1, 2, 3, 3, 3, 3]",
      "label": "in1"
    },
    "p0": {
      "value": "tf.math.segment_max(data, segment_ids)",
      "docstring": "Computes the maximum along segments of a tensor."
    }
  },
  "edges": [
    {
      "start": "p0",
      "end": "n0"
    },
    {
      "start": "n1",
      "end": "p0"
    },
    {
      "start": "n2",
      "end": "p0"
    }
  ],
  "trace_edges": {
    "n0": [
      {
        "start": "n1",
        "is_value_wise": false,
        "description": "",
        "provenance": "[[0], [1, 2], [3], [4, 5, 6, 7]]"
      },
      {
        "start": "n2",
        "is_value_wise": false,
        "description": "",
        "provenance": "[[0], [1, 2], [3], [4, 5, 6, 7]]"
      }
    ]
  }
}
tf.int32:[1, 4, 5, 10]
```

`trace_edges` attributes explanation: 
- each (key, value) pair in `trace_edges` corresponds to a operation's io information. `n0` refers to the output node of the operation `tf.math.segment_max` and `n1`/`n2` refer to its input nodes.

- `is_value_wise`: if false, it means there exists element-wise correspondence between input and output nodes, and we can draw arrows from any elements of the output node to trace its provenance elements in the input nodes. In the example, all values of `n0` can be traced from `n1` and `n2`, therefore both node are `element-wise` instead of `value-wise`. If `true`, it means there is no element-wise correspondence between input and output node, and the whole input node should point directly to the output nodes. For example, if input node is used as `axis` or `shape` argument, it affects the whole output tensor instead of individual elements.

- `provenance`: it is a python sequence, by which we can use the element id of the output node to request a list of ids in the input node.
For example, `n1 -> n0` has `provenance = [[0], [1, 2], [3], [4, 5, 6, 7]]`.
If we want to know where the first element in output node `n0` ,i.e. `1`, comes from, we can use the id of the first element in output node `n0`, i.e. `0`,  to index the provenance, i.e. `provenance[0]`, which gives a list of ids of the input node `n1`, i.e.`[0]`. It means, the first element of the output comes from the first element of the input. If we are curious about where the second element comes from, `provenance[1]` gives `[1,2]`, it means the second element in the output node tensor `n0` is the result of comparing the second and third elements in the input tensor `n1`.

Corner Cases:
- Indexing into the provenance can also gives empty list `[]`, which means this element in the output node is not derived from any elements in the input node.
<!-- - If the output is a primitive instead of a tensor, we do not have the outmost list in provenance. Instead the provenacne itself shows a list of indices without indexing.

```
tensor = [1,2,3] -> reduce_sum(tensor) -> 6`, then provenance = [[0],[1],[2]].

tensor = [[1,2,3]] -> reduce_sum(tensor, axis=-1) -> [6]`, then provenance = [[[0],[1],[2]]]
```

- If the input is a primitive instead of a tensor, for example `0`, `tf.constant(0)` or `tf.dtype`, indexing (via the element id of the output node) into the provenance gives `True` or `False`, instead of a list of indices.

```
tensor=[-1,.5,2], min=1, max=2 -> clip_by_value(tensor, min, max) -> [0,.5,2],
then provenance for min = [True, True, True]
``` -->

- If one node is a primitive instead of a tensor, we regard it as having id 0. 
  - As shown in the example below, if output node is a primitive, we can still use 0 to index the provenance.
  - If input node is a primitive, indexing into provenace gives a list containing only 0, which represents the input primitive.
```
tensor=[1,2,3] -> reduce_sum(tensor) -> 6, then provenance(tensor) = [[0, 1, 2]].

primitive = 2 -> tf.constant(primitive) -> <tf.Tensor: numpy=1>, then provenance(primitive) = [[0]]
```


### More examples
- `google_13`
```
tf.concat((in1, in2), 1)
{
  "nodes": {
    "n0": {
      "type": "intermediate",
      "value": "[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]",
      "label": "tf.concat((in1, in2), 1)"
    },
    "n1": {
      "type": "intermediate",
      "value": "([[1, 2], [10, 20]], [[3, 4, 5], [30, 40, 50]])",
      "label": "(in1, in2)"
    },
    "n2": {
      "type": "input",
      "value": "[[1, 2], [10, 20]]",
      "label": "in1"
    },
    "n3": {
      "type": "input",
      "value": "[[3, 4, 5], [30, 40, 50]]",
      "label": "in2"
    },
    "n4": {
      "type": "constant",
      "value": "1",
      "label": "1"
    },
    "p0": {
      "value": "PairCreationOperation",
      "docstring": "Creates a pair (2-tuple)."
    },
    "p1": {
      "value": "tf.concat(values, axis)",
      "docstring": "Concatenates tensors along one dimension."
    }
  },
  "edges": [
    {
      "start": "p0",
      "end": "n1"
    },
    {
      "start": "n2",
      "end": "p0"
    },
    {
      "start": "n3",
      "end": "p0"
    },
    {
      "start": "p1",
      "end": "n0"
    },
    {
      "start": "n1",
      "end": "p1"
    },
    {
      "start": "n4",
      "end": "p1"
    }
  ],
  "trace_edges": {
    "n0": [
      {
        "start": "n4",
        "is_value_wise": true,
        "description": "",
        "provenance": "None"
      },
      {
        "start": "n1",
        "is_value_wise": false,
        "description": "",
        "provenance": "[[0], [1], [4], [5], [6], [2], [3], [7], [8], [9]]"
      }
    ]
  }
}
tf.int32:[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
```