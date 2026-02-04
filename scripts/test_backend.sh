#!/bin/bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# change work directory to main source directory
cd "$ROOT_DIR"/..

export PYTHONPATH=$PWD/out/pylib:$PWD/pylib
source out/pyenv/bin/activate
python main.py
