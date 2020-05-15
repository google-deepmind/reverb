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

"""Module to check that the TF version is sufficient for Reverb.

 Note: We need to put some imports inside a function call below, and the
 function call needs to come before the *actual* imports that populate the
 reverb namespace. Hence, we disable this lint check throughout the file.
"""
import distutils.version

# pylint: disable=g-import-not-at-top


def ensure_tf_version():
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Note: This function must be executed before Reverb is imported as Reverb will
  attempt to import TensorFlow.

  Raises:
    ImportError: If either tensorflow is not importable or its version is
      inadequate.
  """
  try:
    import tensorflow as tf
  except ImportError:
    print('\n\nFailed to import TensorFlow. Please note that TensorFlow is not '
          'installed by default when you install Reverb. This is so that '
          'users can decide whether to install the GPU-enabled TensorFlow '
          'package. To use Reverb, please install the most recent version '
          'of TensorFlow, by following instructions at '
          'https://tensorflow.org/install.\n\n')
    raise

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  required_tensorflow_version = '2.3.0'

  version = tf.version.VERSION
  if (distutils.version.LooseVersion(version) <
      distutils.version.LooseVersion(required_tensorflow_version)):
    raise ImportError(
        'This version of Reverb requires TensorFlow '
        'version >= {required}; Detected an installation of version {present}. '
        'Please upgrade TensorFlow to proceed.'.format(
            required=required_tensorflow_version,
            present=version))

