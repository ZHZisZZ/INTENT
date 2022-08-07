import time
from collections import defaultdict
from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.benchmarks import benchmark as benchmark_module
from tf_coder.dataflow import dataflow
from INTENT import session

WAITTIME = 20 * 60
UPWEIRGHT = 3


def get_inputs_and_output_list(request_data):
    inputs_list, output_list = [], []

    for inputs, output in \
            zip(request_data['inputs'].values(),
                request_data['outputs'].values()):
        inputs_list.append(list(map(
            lambda i: eval(i), inputs.values())))
        output_list.append(eval(output['1']))

    return inputs_list, output_list


def get_settings(desired_op, undesired_op, time_limit,
                 number_of_solutions, **argvs):
    """Specifies settings for TF-Coder. Edit this function!"""

    return settings_module.from_dict({
        'timeout': time_limit,
        'max_extra_solutions_time': time_limit,
        'only_minimal_solutions': False,
        'max_solutions': number_of_solutions,
        'require_all_inputs_used': True,
        'require_one_input_used': False,

        'operations.undesired_multiplier': UPWEIRGHT,
        'operations.desired_multiplier': 1 / UPWEIRGHT,
        'operations.desired_operations': desired_op,
        'operations.undesired_operations': undesired_op
    })


def solve_problem(inputs_list, output_list, constants,
                  description, session_id, **argvs):
    """Specifies a problem to run TF-Coder on. Edit this function!"""
    session_info = session.global_session_dict[session_id]
    if not colab_interface.WARMED_UP:
        colab_interface.warm_up()
    settings = get_settings(**argvs)
    colab_interface.run_value_search_from_colab(
        inputs_list, output_list, constants,
        description, settings, session_info.asyn)
    # session_info.response.completed = True
    session_info.solver_completed = True
    # clear session after waiting for 20 minutes
    time.sleep(WAITTIME)
    session.global_session_dict.pop(session_id)

 
def validate_solutions(request_data, asyn_solutions):
    response = defaultdict(defaultdict)

    inputs_list, output_list = get_inputs_and_output_list(request_data)
    text_solutions = request_data['solutions'].values()

    asyn_text_solutions = [solution.expression for solution in asyn_solutions]

    for i_example, (inputs, output) in enumerate(zip(inputs_list, output_list)):

        for i_solution, text_solution in enumerate(text_solutions):

            
            match, exception, eval_output = value_search._check_application(
                text_solution, benchmark_module.Example(inputs, output))

            # exception occurs during eval()
            if exception:
                response[f'example{i_example}'][f'solution{i_solution}'] = \
                    session.Validation(
                        match, exception, eval_output,
                        session.Graph(None, None, None)._asdict())._asdict()
                continue

            # text_solution is searched by tf-coder before
            value = None
            if text_solution in asyn_text_solutions:
                index = asyn_text_solutions.index(text_solution)
                value = asyn_solutions[index].value
                value = dataflow.value_from_different_inputs(value, inputs)
            else:  # text_solution is provided by user
                try:
                    value = dataflow.value_from_text(text_solution, inputs)
                except Exception as e:
                    match = False
                    exception = True
                    eval_output = str(e)
                    response[f'example{i_example}'][f'solution{i_solution}'] = \
                        session.Validation(
                            match, exception, eval_output,
                            session.Graph(None, None, None)._asdict())._asdict()
                    continue

            nodes, edges, trace_edges = \
                dataflow.dataflow_generator(value)

            response[f'example{i_example}'][f'solution{i_solution}'] = \
                session.Validation(
                    match, exception, eval_output,
                    session.Graph(nodes, edges, trace_edges)._asdict())._asdict()

    return response        
        