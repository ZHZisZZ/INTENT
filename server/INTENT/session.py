from typing import Dict, Text, Union, Tuple
from collections import defaultdict, namedtuple
from tf_coder import asyn_utils

Solution = namedtuple('Solution', ['expression', 'weight', 'time'])
Graph = namedtuple('Graph', ['nodes', 'edges', 'trace_edges'])

Validation = namedtuple('Validation', ['match', 'exception', 'eval_output', 'graphs'])


class Response(object):

    def __init__(self):
        self.completed = False
        self.solutions = dict()
        self.graphs = defaultdict(dict)

    def _asdict(self):
        return {
            'completed': self.completed,
            'solutions': self.solutions,
            'graphs': self.graphs
        }


class Session(object):

    def __init__(self):
        self.asyn = asyn_utils.Asyn()
        self.solver_completed = False
        self.response = Response()  # /poll response
        self.inputs_list = []


global_session_dict: Dict[int, Session] = defaultdict(Session)
