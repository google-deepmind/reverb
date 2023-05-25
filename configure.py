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
"""Configure script to get build parameters from user.

This should be run before building reverb with Bazel. The easiest usage is
`python3 configure.py`. It will use the version of python to suggest the correct
paths to set for the bazel config.

Shamelessly taken from TensorFlow:
  htps://github.com/tensorflow/tensorflow/blob/master/configure.py
"""
import argparse
import os
import subprocess
import sys
import platform
import tensorflow as tf

_REVERB_BAZELRC_FILENAME = '.reverb.bazelrc'
_REVERB_WORKSPACE_ROOT = ''
_REVERB_BAZELRC = ''


def main():
  global _REVERB_WORKSPACE_ROOT
  global _REVERB_BAZELRC

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default=os.path.abspath(os.path.dirname(__file__)),
      help='The absolute path to your active Bazel workspace.')
  parser.add_argument(
      '--force_defaults',
      type=bool,
      default=False,
      help='Whether to force the usage of default values, skipping manual selection. This can be useful for automated scripts'
  )
  args = parser.parse_args()

  _REVERB_WORKSPACE_ROOT = args.workspace
  _REVERB_BAZELRC = os.path.join(_REVERB_WORKSPACE_ROOT,
                                 _REVERB_BAZELRC_FILENAME)

  # Make a copy of os.environ to be clear when functions and getting and setting
  # environment variables.
  environ_cp = dict(os.environ)

  reset_configure_bazelrc()
  setup_python(environ_cp, args.force_defaults)


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var,
                                    var_default):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = _get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def _get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def setup_python(environ_cp, force_defaults: bool):
  """Setup python related env variables.

  Args:
    environ_cp: The environment dict, as `dict(os.environ)`.
    force_defaults: If True, we won't be asking for user inputs. Suited for
      scripts.
  """
  # Get PYTHON_BIN_PATH, default is the current running python.
  default_python_bin_path = sys.executable
  ask_python_bin_path = ('Please specify the location of python. [Default is '
                         '%s]: ') % default_python_bin_path
  while True:
    if force_defaults:
      python_bin_path = default_python_bin_path
      print(f'Using Python binary {python_bin_path}')
    else:
      python_bin_path = get_from_env_or_user_or_default(
          environ_cp, 'PYTHON_BIN_PATH', ask_python_bin_path,
          default_python_bin_path)
    # Check if the path is valid
    if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
      break
    elif not os.path.exists(python_bin_path):
      print('Invalid python path: %s cannot be found.' % python_bin_path)
      if force_defaults:
        return
    else:
      print('%s is not executable.  Is it the python binary?' % python_bin_path)
      if force_defaults:
        return
    environ_cp['PYTHON_BIN_PATH'] = ''

  # Get PYTHON_LIB_PATH
  python_lib_path = environ_cp.get('PYTHON_LIB_PATH')
  if not python_lib_path:
    python_lib_paths = get_python_path(environ_cp, python_bin_path)
    if environ_cp.get('USE_DEFAULT_PYTHON_LIB_PATH') == '1':
      python_lib_path = python_lib_paths[0]
    else:
      print('Found possible Python library paths:\n  %s' %
            '\n  '.join(python_lib_paths))
      default_python_lib_path = python_lib_paths[0]
      if force_defaults:
        python_lib_path = default_python_lib_path
        print(f'Using Python lib {default_python_lib_path}')
      else:
        python_lib_path = _get_input(
            'Please input the desired Python library path to use.  '
            'Default is [%s]\n' % default_python_lib_path)
        if not python_lib_path:
          python_lib_path = default_python_lib_path
    environ_cp['PYTHON_LIB_PATH'] = python_lib_path

  # Set-up env variables used by python_configure.bzl
  write_action_env_to_bazelrc('PYTHON_BIN_PATH', python_bin_path)
  write_action_env_to_bazelrc('PYTHON_LIB_PATH', python_lib_path)
  write_to_bazelrc('build --python_path=\"%s"' % python_bin_path)
  write_to_bazelrc('build --repo_env=PYTHON_BIN_PATH=\"%s"' % python_bin_path)
  environ_cp['PYTHON_BIN_PATH'] = python_bin_path

  write_to_bazelrc('build --spawn_strategy=standalone')
  write_to_bazelrc('build --strategy=Genrule=standalone')

  # If choosen python_lib_path is from a path specified in the PYTHONPATH
  # variable, need to tell bazel to include PYTHONPATH
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
    if python_lib_path in python_paths:
      write_action_env_to_bazelrc('PYTHONPATH', environ_cp.get('PYTHONPATH'))

  cflags = tf.sysconfig.get_compile_flags()
  lflags = tf.sysconfig.get_link_flags()

  tf_header_dir = next(cflag[2:] for cflag in cflags if cflag.startswith('-I'))
  tf_shared_dir = next(lflag[2:] for lflag in lflags if lflag.startswith('-L'))
  tf_shared_lib_name = next(lflag[2:] for lflag in lflags if lflag.startswith('-l'))
  lib_ext = {
    'Darwin': 'dylib',
    'Linux': 'so',
  }[platform.system()]
  tf_lib_prefix = 'lib' if platform.system() in ('Darwin', 'Linux') else ''
  
  write_action_env_to_bazelrc('TF_HEADER_DIR', tf_header_dir)
  write_action_env_to_bazelrc('TF_SHARED_LIBRARY_DIR', tf_shared_dir)
  write_action_env_to_bazelrc('TF_SHARED_LIBRARY_NAME', tf_lib_prefix + tf_shared_lib_name + os.path.extsep + lib_ext)

  # if platform.system() == 'Darwin':
  #   write_to_bazelrc('build --copt="-undefined dynamic_lookup"')

  # Write tools/python_bin_path.sh
  with open(os.path.join(_REVERB_WORKSPACE_ROOT, 'python_bin_path.sh'),
            'w') as f:
    f.write('export PYTHON_BIN_PATH="%s"' % python_bin_path)


def get_python_path(environ_cp, python_bin_path):
  """Get the python site package paths."""
  python_paths = []
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
  try:
    stderr = open(os.devnull, 'wb')
    library_paths = run_shell([
        python_bin_path, '-c',
        'import site; print("\\n".join(site.getsitepackages()))'
    ],
                              stderr=stderr).split('\n')
  except subprocess.CalledProcessError:
    library_paths = [
        run_shell([
            python_bin_path, '-c',
            ('from distutils.sysconfig import get_python_lib;'
             'print(get_python_lib())')
        ])
    ]

  all_paths = set(python_paths + library_paths)

  paths = []
  for path in all_paths:
    if os.path.isdir(path):
      paths.append(path)
  return paths


def run_shell(cmd, allow_non_zero=False, stderr=None):
  """Get var_name either from env, or user or default.

  Args:
    cmd: copy of the os.environ.
    allow_non_zero: string for name of environment variable, e.g. "TF_NEED
    stderr: string for how to ask for user input.

  Returns:
    string value output of the command executed.
  """
  if stderr is None:
    stderr = sys.stdout
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd, stderr=stderr)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd, stderr=stderr)
  return output.decode('UTF-8').strip()


def write_action_env_to_bazelrc(var_name, var):
  write_to_bazelrc('build --action_env %s="%s"' % (var_name, str(var)))


def write_to_bazelrc(line):
  with open(_REVERB_BAZELRC, 'a') as f:
    f.write(line + '\n')


def reset_configure_bazelrc():
  """Reset file that contains customized config settings."""
  open(_REVERB_BAZELRC, 'w').close()


if __name__ == '__main__':
  main()
