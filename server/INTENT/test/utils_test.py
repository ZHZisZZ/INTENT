import json
from tf_coder import asyn_utils
from tf_coder.value_search import colab_interface, value_search_settings

from INTENT import utils


if __name__ == '__main__':
    
    # case 1
    # parse solution string from user input
    request_data = \
        {
            'inputs': {'1': {'1': '[[1,2],[3,4]]'}, '2': {'1': '[[3,4],[5,6]]'}},
            'outputs': {'1': {'1': '[[1,3],[2,4]]'}, '2': {'1': '[[0,0],[0,0]]'}},
            'solutions': {
                'solution0': 'tf.transpose(in1)',
                'solution1': 'tf.cast(tf.transpose(tf.cast(in1, tf.float32)), tf.int32)',
                'solution2': 'tf.argsort(in1, in1)'
                }
        }
    
    res = utils.validate_solutions(request_data, [])
    print(json.dumps(request_data, indent=4))
    print(json.dumps(res, indent=4))
    
    # case 2
    # first find the solution in the cache, if can not find
    # parse the input solution string
    # asyn = asyn_utils.Asyn()
    # inputs_list, output_list = utils.get_inputs_and_output_list(request_data)
    # colab_interface.warm_up()
    # colab_interface.run_value_search_from_colab(
    #     inputs_list[:-1], output_list[:-1], [], '', 
    #     value_search_settings.default_settings(), asyn)
    # asyn_solutions = asyn.solutions
    # print(asyn_solutions)
    # res = utils.validate_solutions(request_data, asyn_solutions)
    # print(res)