#!/bin/bash
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Executes Reverb Python tests.
#
# Expects Reverb to be installed and PYTHON_BIN_PATH to be set to the path of
# python to use.

# Usage (expects Reverb to be installed):
#   export PYTHON_BIN_PATH=$(which python3)
#   bash run_python_tests.sh
set +x

py_test() {
  local exit_code=0

  echo "===========Running Python tests============"

  cd reverb/  # Fix OSX circular import error
  for test_file in `find ./ -name '*_test.py' -print`
  do
    echo "####=======Testing ${test_file}=======####"
    ${PYTHON_BIN_PATH} "${test_file}"
    _exit_code=$?
    if [[ $_exit_code != 0 ]]; then
      exit_code=$_exit_code
      echo "FAIL: ${test_file}"
    fi
  done

  return "${exit_code}"
}

py_test
