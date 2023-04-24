#!/bin/bash
set -e

# The first argument is the name of the venv
VENV_NAME=$1
# The remainder are passed to the python executable
ARGS=${@:2}

# Change into the directory containing this script
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$SCRIPTPATH"

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    # For M1, preload libgomp to avoid this bug https://stackoverflow.com/questions/67735216/after-using-pip-i-get-the-error-scikit-learn-has-not-been-built-correctly
    LD_PRELOAD=libgomp.so.1 PATH=$PATH:/app/$VENV_NAME/.venv/bin /app/$VENV_NAME/.venv/bin/python3 -u $ARGS
else
    # Run using python executable from the provided venv
    PATH=$PATH:/app/$VENV_NAME/.venv/bin /app/$VENV_NAME/.venv/bin/python3 -u $ARGS
fi
