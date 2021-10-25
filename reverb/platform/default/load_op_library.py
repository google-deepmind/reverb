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

"""Helpers for loading dynamic library files."""
import sys

import six
import tensorflow as tf

UNDEFINED_SYMBOL_ERROR_MESSAGE = """
Attempted to load a reverb dynamic library, but it could not find the required
symbols inside of TensorFlow.  This commonly occurs when your version of
tensorflow and reverb are mismatched.  For example, if you are using the python
package 'tf-nightly', make sure you use the python package 'dm-reverb-nightly'
built on the same or the next night.  If you are using a release version package
'tensorflow', use a release package 'dm-reverb' built to be compatible with
that exact version.  If all else fails, file a github issue on deepmind/reverb.
Current installed version of tensorflow: {tf_version}.
""".format(tf_version=tf.__version__)


def reraise_wrapped_error(error: Exception):
  """Wraps failures with better error messages.

  Args:
    error: The exception.  We must be inside a raise.

  Raises:
    ImportError: Typically if there is a version mismatch.
  """
  if 'undefined symbol' in str(error).lower():
    six.reraise(ImportError,
                ImportError('%s\nOrignal error:\n%s' % (
                    UNDEFINED_SYMBOL_ERROR_MESSAGE, error)),
                sys.exc_info()[2])
  raise error
