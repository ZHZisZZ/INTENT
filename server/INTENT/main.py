"""
INTENT index (main) view.

URLs include:
/
"""
import flask
import flask_cors
import threading


import INTENT
from INTENT import utils, session
from tf_coder.dataflow import dataflow
from datetime import timezone, datetime


@INTENT.app.route('/', methods=['GET'])
@INTENT.app.route('/<version>', methods=['GET'])
def index(version='beta'):
    """Display / route."""
    if version in ['alpha', 'beta']:
        return flask.render_template("index.html")
    flask.abort(404)


@INTENT.app.errorhandler(404)
def page_not_found(e):
    html = f"""<a href="{flask.url_for('index')}"><p>
               Click Here</a> To go to the Home Page</p>"""
    return html, 404


@INTENT.app.route('/solve',
                     methods=['POST', 'OPTIONS', 'HEAD'])
@flask_cors.cross_origin()
def solve():
    session_id = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    data = flask.request.json

    inputs_list, output_list = utils.get_inputs_and_output_list(data)

    session.global_session_dict[session_id].inputs_list = inputs_list

    # worker thread in background
    solver = threading.Thread(
        target=utils.solve_problem,
        kwargs={'inputs_list': inputs_list,
                'output_list': output_list,
                'constants': eval(data['constraints']),
                'description': data['description'],
                'desired_op': list(data['desired_op'].keys()),
                'undesired_op': list(data['undesired_op'].keys()),
                'time_limit': float(data['timeout']),
                'number_of_solutions': int(data['sol_num']),
                'session_id': session_id})

    solver.start()

    return flask.jsonify({'session_id': session_id})


@INTENT.app.route('/poll/<int:session_id>',
                     methods=['GET'])
@flask_cors.cross_origin()
def poll(session_id):
    """
    response:
    {        
        'completed': bool,
        'solutions': {
            'solution0': {
                'expression': str,
                'weight': int,
                'time': double,    
            },
            'solution1': {...},
            ...,
        }
        'graphs': {
            'example0': {
                'solution0': {
                    'nodes': {...},
                    'edges': {...},
                    'trace_edges': {...},   
                }
                'solution1': {...},
                ...,
            },
            'example1': {...},
            ...,
        }
    }
    """
    session_info = session.global_session_dict[session_id]
    solutions = session_info.response.solutions
    graphs = session_info.response.graphs
    inputs_list = session_info.inputs_list
    n_unprocessed = len(session_info.asyn.solutions)
    n_processed = len(session_info.response.solutions)
    
    session_info.response.completed = session_info.solver_completed

    # conduct data provenance trace when new unprocessed solutions are found
    for i in range(n_processed, n_unprocessed):
        unprocessed_solution = \
            session_info.asyn.solutions[i]

        solutions[f'solution{i}'] = \
            session.Solution(unprocessed_solution.expression,
                             unprocessed_solution.weight,
                             round(unprocessed_solution.time, 2))._asdict()

        for j, inputs in enumerate(inputs_list):
            value = dataflow.value_from_different_inputs(
                    unprocessed_solution.value, inputs)

            nodes, edges, trace_edges = \
                dataflow.dataflow_generator(value)

            graphs[f'example{j}'][f'solution{i}'] = \
                session.Graph(nodes, edges, trace_edges)._asdict()

    # print(session_info.response.completed)
    return flask.jsonify(session_info.response._asdict())


@INTENT.app.route('/abort/<int:session_id>',
                     methods=['GET'])
@flask_cors.cross_origin()
def abort(session_id):
    # aborting an unexsiting session_id has no effects
    if session_id in session.global_session_dict:
        session_info = session.global_session_dict[session_id]
        session_info.asyn.aborted = True
    # colab_interface.run_value_search_from_colab (in solver thread)
    # will terminate immediately and session_id will be poped after 5s
    return ''


@INTENT.app.route('/validate',
                     methods=['POST', 'OPTIONS', 'HEAD'])
@flask_cors.cross_origin()
def validate():
    """
    request:
    {
        'inputs': {...}, // the same structure as in /solve
        'outputs': {...}, // the same structure as in /solve
        'solutions': {'solution0': str, ...}, 
    }

    response:
    {        
        'example0': {
            'solution0': {
                'match': bool,
                'exception': bool,
                'eval_output': exception message or new output,
                'graphs': {
                    'nodes': {...}, # null if exception is true
                    'edges': {...}, # null if exception is true
                    'trace_edges': {...}, # null if exception is true
                }
            }
            'solution1': {...},
            ...,
        },
        'example1': {...},
    }
    """
    request_data = flask.request.json
    if 'session_id' not in request_data:
        asyn_solutions = []
    else:
        asyn_solutions = session.global_session_dict[request_data['session_id']].asyn.solutions

    # print(utils.validate_solutions(request_data, asyn_solutions))
    return flask.jsonify(utils.validate_solutions(request_data, asyn_solutions))


################################################################################
@INTENT.app.route('/show_dict', methods=['GET'])
def show_dict():
    """Get current global dict status (only for test)"""
    return str(session.global_session_dict)

@INTENT.app.route('/clear_dict', methods=['GET'])
def clear_dict():
    """Get current global dict status (only for test)"""
    return str(session.global_session_dict)
