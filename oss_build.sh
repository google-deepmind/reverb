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

# Designed to work with ./docker/release.dockerfile to build reverb for multiple
# Python versions. It can work locally but is not tested for that use case.
#
# Example usage after building release docker:
#   docker run --rm -it -v ${REVERB_DIR}:/tmp/reverb tensorflow:reverb_release \
#   bash oss_build.sh --python 3.8

# Exit if any process returns non-zero status.
set -e
set -o pipefail

# Flags
PYTHON_VERSIONS=3.7 # Options 3.7 (default), 3.8, or 3.9.
CLEAN=false # Set to true to run bazel clean.
OUTPUT_DIR=/tmp/reverb/dist/
PYTHON_TESTS=true
DEBUG_BUILD=false

ABI=cp36
PIP_PKG_EXTRA_ARGS="" # Extra args passed to `build_pip_package`.

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--release [Indicates this is a release build. Otherwise nightly.]"
  echo "--python [3.7(default)|3.8|3.9|3.10]"
  echo "--clean  [true to run bazel clean]"
  echo "--tf_dep_override  [Required tensorflow version to pass to setup.py."
  echo "                    Examples: tensorflow==2.3.0rc0  or tensorflow>=2.3.0]"
  echo "--python_tests  [true (default) to run python tests.]"
  echo "--output_dir  [location to copy .whl file.]"
  echo "--debug_build  [true to build a debug binary.]"
  exit 1
fi

while [[ $# -gt -0 ]]; do
  key="$1"
  case $key in
      --release)
      PIP_PKG_EXTRA_ARGS="${PIP_PKG_EXTRA_ARGS} --release" # Indicates this is a release build.
      ;;
      --python)
      PYTHON_VERSIONS="$2" # Python versions to build against.
      shift
      ;;
      --clean)
      CLEAN="$2" # `true` to run bazel clean. False otherwise.
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
      --debug_build)
      DEBUG_BUILD="$2"
      shift
      ;;
      --tf_dep_override)
      # Setup.py is told this is the tensorflow dependency.
      PIP_PKG_EXTRA_ARGS="${PIP_PKG_EXTRA_ARGS} --tf-version ${2}"
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      exit 1
      ;;
  esac
  shift # past argument or value
done

for python_version in $PYTHON_VERSIONS; do

  # Cleans the environment.
  if [ "$CLEAN" = "true" ]; then
    bazel clean
  fi

  if [ "$python_version" = "3.7" ]; then
    ABI=cp37
  elif [ "$python_version" = "3.8" ]; then
    ABI=cp38
  elif [ "$python_version" = "3.9" ]; then
    ABI=cp39
  elif [ "$python_version" = "3.10" ]; then
    ABI=cp310
  else
    echo "Error unknown --python. Only [3.7|3.8|3.9|3.10]"
    exit 1
  fi

  export PYTHON_BIN_PATH=`which python${python_version}`
  export PYTHON_LIB_PATH=`${PYTHON_BIN_PATH} -c 'import site; print(site.getsitepackages()[0])'`

  if [ "$(uname)" = "Darwin" ]; then
    bazel_config=""
    PLATFORM=`${PYTHON_BIN_PATH} -c "from distutils import util; print(util.get_platform())"`
  else
    bazel_config="--config=manylinux2014"
    PLATFORM="manylinux2014_x86_64"
  fi

  # Configures Bazel environment for selected Python version.
  $PYTHON_BIN_PATH configure.py

  # Runs bazel tests for cc.
  # Only run cc tests because `bazel test` seems to ignore bazelrc and only uses
  # /usr/bin/python3. A solution is to swap symbolic links for each version of
  # python to be tested. This works well in docker but would make a mess of
  # someone's system unexpectedly. We are executing the python tests after
  # installing the final package making this approach satisfactory.
  # TODO(b/157223742): Execute Python tests as well.
  bazel test -c opt --copt=-mavx ${bazel_config} --test_output=errors //reverb/cc/...

  EXTRA_OPT=""
  if [ "$DEBUG_BUILD" = "true" ]; then
     EXTRA_OPT="--copt=-g2"
  fi

  # Builds Reverb and creates the wheel package.
  bazel build -c opt --copt=-mavx $EXTRA_OPT $bazel_config reverb/pip_package:build_pip_package
  ./bazel-bin/reverb/pip_package/build_pip_package --dst $OUTPUT_DIR $PIP_PKG_EXTRA_ARGS --platform "$PLATFORM"

  # Installs pip package.
  $PYTHON_BIN_PATH -m pip install --force-reinstall ${OUTPUT_DIR}*${ABI}*.whl

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
