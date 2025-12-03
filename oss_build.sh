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

# Example usage after building release docker:
#   bash oss_build.sh --python 3.9

# Exit if any process returns non-zero status.
set -e
set -o pipefail

# Flags
PYTHON_VERSIONS=3.11 # Options 3.9 (default), 3.10 or 3.11.
OUTPUT_DIR=wheelhouse
PYTHON_TESTS=true
WHEEL_NAME=dm_reverb_nightly
WHEEL_TYPE=nightly
BUILD_DATE=`date '+%Y%m%d'`
export ML_WHEEL_NAME=$WHEEL_NAME
BAZEL_ARGS="--repo_env=WHEEL_NAME=$WHEEL_NAME --repo_env=ML_WHEEL_TYPE=$WHEEL_TYPE --repo_env=ML_WHEEL_BUILD_DATE=$BUILD_DATE"
BAZEL_BIN="${BAZEL_BIN:-bazel}"

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--release [Indicates this is a release build. Otherwise nightly.]"
  echo "--python [3.9(default)|3.10|3.11]"
  echo "--clear_bazel_cache  [true to delete Bazel cache folder]"
  echo "--tf_dep_override  [Required tensorflow version to pass to setup.py."
  echo "                    Examples: tensorflow==2.3.0rc0  or tensorflow>=2.3.0]"
  echo "--output_dir  [location to copy .whl file.]"
  exit 1
fi

while [[ $# -gt -0 ]]; do
  key="$1"
  case $key in
      --release)
      export WHEEL_NAME=dm_reverb
      BAZEL_ARGS="--repo_env=WHEEL_NAME=$WHEEL_NAME --repo_env=ML_WHEEL_TYPE=release"
      ;;
      --python)
      PYTHON_VERSIONS="$2" # Python versions to build against.
      shift
      ;;
      --python_tests)
      PYTHON_TESTS="$2"
      shift
      ;;
      --output_dir)
      OUTPUT_DIR="$2"
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      exit 1
      ;;
  esac
  shift # past argument or value
done

echo BAZEL_ARGS=$BAZEL_ARGS

for python_version in $PYTHON_VERSIONS; do

  $BAZEL_BIN test -c opt --repo_env=HERMETIC_PYTHON_VERSION=$python_version $BAZEL_ARGS //reverb/...

  # Builds Reverb and creates the wheel package.
  output_wheel=$($BAZEL_BIN cquery --repo_env=HERMETIC_PYTHON_VERSION=$python_version $BAZEL_ARGS --output=files //reverb/pip_package:wheel 2> /dev/null)
  $BAZEL_BIN build -c opt --repo_env=HERMETIC_PYTHON_VERSION=$python_version $BAZEL_ARGS //reverb/pip_package:wheel
  echo "Created $output_wheel"

  mkdir -p $OUTPUT_DIR/

  if [ "$(uname -s)" == "Darwin" ]; then
    install_wheel=$OUTPUT_DIR/$(basename $output_wheel)
    cp $output_wheel $OUTPUT_DIR/$(basename $output_wheel)
  else
    platform=linux_x86_64
    target_platform=manylinux_2_27_x86_64
    uvx auditwheel repair \
      --plat $target_platform \
      --exclude libtensorflow_framework.so.2 \
      --wheel-dir $OUTPUT_DIR \
      $output_wheel

    install_wheel=$OUTPUT_DIR/"$(basename $output_wheel | sed "s/$platform/$target_platform/")"
  fi

  # Installs pip package.
  uv venv --clear --python $python_version --seed ./venvs/py$python_version
  export PYTHON_BIN_PATH="./venvs/py$python_version/bin/python3"
  $PYTHON_BIN_PATH -mpip install "$WHEEL_NAME[tensorflow] @ file:$install_wheel"

  if [ "$PYTHON_TESTS" = "true" ]; then
    echo "Run Python tests..."
    set +e

    bash run_python_tests.sh 2>&1 | tee ./unittest_log.txt
    UNIT_TEST_ERROR_CODE=$?
    set -e
    if [[ $UNIT_TEST_ERROR_CODE != 0 ]]; then
      echo -e "\n\n\n===========Error Summary============"
      grep -E 'ERROR:|FAIL:' ./unittest_log.txt
      exit $UNIT_TEST_ERROR_CODE
    else
      echo "Python tests successful!!!"
    fi
  fi

done
