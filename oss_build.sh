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
PYTHON_VERSIONS=3.11 # [3.9|3.10|3.11|3.12|3.13]
OUTPUT_DIR=dist
PYTHON_TESTS=true
BUILD_DATE=`date '+%Y%m%d'`
RELEASE=false
BAZEL_BIN="${BAZEL_BIN:-bazel}"

if ! command -v uv >/dev/null 2>&1
then
    echo "Building reverb requires uv."
    exit 1
fi

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--release [true|false, indicates this is a release build. Otherwise nightly.]"
  echo "--python [3.9|3.10|3.11|3.12|3.13]"
  echo "--output_dir  [location to copy .whl file.]"
  echo "--python_tests  [true|false, whether to run python tests with built wheel]"
  exit 1
fi

while [[ $# -gt -0 ]]; do
  key="$1"
  case $key in
      --release)
      RELEASE=true
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

mkdir -p "$OUTPUT_DIR/"

for python_version in $PYTHON_VERSIONS; do

  BAZEL_ARGS="--repo_env=HERMETIC_PYTHON_VERSION=$python_version"
  if [ "$RELEASE" == "true" ]; then
    BAZEL_ARGS="$BAZEL_ARGS --repo_env=WHEEL_NAME=dm_reverb --repo_env=ML_WHEEL_TYPE=release"
  else
    BAZEL_ARGS="$BAZEL_ARGS --repo_env=WHEEL_NAME=dm_reverb_nightly --repo_env=ML_WHEEL_TYPE=nightly --repo_env=ML_WHEEL_BUILD_DATE=$BUILD_DATE"
  fi

  $BAZEL_BIN test $BAZEL_ARGS //reverb/...
  # Builds Reverb and creates the wheel package.
  output_wheel=$($BAZEL_BIN cquery $BAZEL_ARGS --output=files //reverb/pip_package:wheel 2> /dev/null)
  $BAZEL_BIN build $BAZEL_ARGS //reverb/pip_package:wheel

  # Only macOS ARM and linux x86 are currently supported.
  if [ "$(uname -s)" == "Darwin" ]; then
    # You may see error messages of the form
    # @rpath/libtensorflow_framework.2.dylib not found:
    # They are safe to discard as libtensorflow_framework.2.dylib will be resolved
    # at runtime to the installed tensorflow package.
    MACOSX_DEPLOYMENT_TARGET=12.0 uvx --from delocate delocate-wheel --ignore-missing-dependencies --wheel-dir $OUTPUT_DIR $output_wheel

    install_wheel="$OUTPUT_DIR/$(basename $output_wheel)"
  else
    platform=linux_x86_64
    target_platform=manylinux_2_27_x86_64
    uvx auditwheel repair --plat $target_platform --exclude libtensorflow_framework.so.2 --wheel-dir $OUTPUT_DIR $output_wheel

    install_wheel=$OUTPUT_DIR/"$(basename $output_wheel | sed "s/$platform/$target_platform/")"
  fi

  if [ "$PYTHON_TESTS" = "true" ]; then
    # Installs pip package.
    uv venv --clear --python $python_version ./venvs/py$python_version
    source ./venvs/py$python_version/bin/activate

    uv pip install "$install_wheel[tensorflow]"

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

    deactivate
  fi

done
