# Lint as: python3
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

"""Data structures for output of client samples."""

from typing import Any, NamedTuple, Sequence, Union

import numpy as np
import tensorflow.compat.v1 as tf


class _SampleInfo(NamedTuple):
  """Extra details about the sampled item.

  Fields:
    key: Key of the item that was sampled. Used for updating the priority.
      Typically a python `int` (for output of Client.sample) or
      `tf.uint64` Tensor (for output of TF Client.sample).
    probability: Probability of selecting the item at the time of sampling.
      A python `float` or `tf.float64` Tensor.
    table_size: The total number of items present in the table at sample time.
    priority: Priority of the item at the time of sampling. A python `float` or
      `tf.float64` Tensor.
  """
  key: Union[np.ndarray, tf.Tensor]
  probability: Union[np.ndarray, tf.Tensor]
  table_size: Union[np.ndarray, tf.Tensor]
  priority: Union[np.ndarray, tf.Tensor]

  @classmethod
  def tf_dtypes(cls):
    return cls(tf.uint64, tf.double, tf.int64, tf.double)


# This makes the value of the field `priority` return with the value of
# `probability` when `priority` is not defined. This is needed to make test pass
# which expects this defined (e.g. when interleave dataset with another one
# containing defined priorities).
# TODO(b/156414572): Remove this once the change is available in nightly.
class SampleInfo(_SampleInfo):
  """SampleInfo which sets priority to probability if it is not provided."""

  def __new__(cls,
              key: Union[np.ndarray, tf.Tensor],
              probability: Union[np.ndarray, tf.Tensor],
              table_size: Union[np.ndarray, tf.Tensor],
              priority: Union[None, np.ndarray, tf.Tensor] = None):
    if priority is None:
      return super(SampleInfo, cls).__new__(
          cls, key, probability, table_size, priority=probability)
    return super(SampleInfo, cls).__new__(
        cls, key, probability, table_size, priority=priority)


class ReplaySample(NamedTuple):
  """Item returned by sample operations.

  Fields:
    info: Details about the sampled item. Instance of `SampleInfo`.
    data: Tensors for the data. Flat list of numpy arrays for output of python
      `Client.sample`, nested structure of Tensors for TensorFlow
      `Client.sample`.
  """
  info: SampleInfo
  data: Union[Sequence[np.ndarray], Any]
