#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

set -ex

function build_wheel() {
  TMPDIR="$1"
  # DEST="$2"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e tools/python_bin_path.sh ]]; then
    source tools/python_bin_path.sh
  fi
  PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-$(which python3)}

  pushd ${TMPDIR} > /dev/null

  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel > /dev/null
  DEST=${TMPDIR}/dist/
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

function prepare_src() {
  TMPDIR="${1%/}"
  mkdir -p "$TMPDIR"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  RUNFILES=bazel-bin/reverb/pip_package/build_pip_package.runfiles/reverb

  # TODO(b/155307832): Copy LICENSE file(s).
  cp -L -R ${RUNFILES}/reverb ${TMPDIR}/reverb

  cp ${TMPDIR}/reverb/pip_package/setup.py ${TMPDIR}
  cp ${TMPDIR}/reverb/pip_package/MANIFEST.in ${TMPDIR}

  # TODO(b/155300149): Don't move .so files to the top-level directory.
  # This copies all .so files except for those found in the ops directory, which
  # must remain where they are for TF to find them.
  find "${TMPDIR}/reverb/cc" -type d -name ops -prune -o -name '*.so' \
    -exec mv {} "${TMPDIR}/reverb" \;
}

function main() {
  TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"

  if [ ! -d bazel-bin/reverb ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  prepare_src "$TMPDIR"
  build_wheel "$TMPDIR"
}

main "$@"
