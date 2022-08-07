#!/bin/bash
# INTENTRun

DEFAULT_PORT=8000

usage()
{
  echo "Usage: $0 [ -p PORT ]"
  exit 2
}

while getopts 'p:?h' c
do
  case $c in
    p) PORT=$OPTARG ;;
    h|?) usage ;; esac
done

if [ -z "$PORT" ]; then
  PORT=$DEFAULT_PORT
fi

# Stop on errors
# See https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
# set -Eeuo pipefail

# Add tf-coder package path to PYTHONPATH package search path
# export PYTHONPATH=$( cd $(git rev-parse --show-cdup); pwd)/tensorflow-coder
BIN_DIR=$(dirname $0)
export PYTHONPATH="$BIN_DIR/../../tensorflow-coder"

# Set FLASK_ENV and FLASK_APP environment variables
export FLASK_ENV=development
FLASK_ENV=development

export FLASK_APP=INTENT
FLASK_APP=INTENT

# Run the development server on port 8000
flask run --host 0.0.0.0 --port $PORT