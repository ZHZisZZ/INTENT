"""INTENT package initializer."""
import flask
from flask_cors import CORS

# app is a single object used by all the code modules in this package
app = flask.Flask(__name__, static_url_path="/server/INTENT/static")  # pylint: disable=invalid-name

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
 
# suppress tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU is faster than GPU.

import INTENT.main
