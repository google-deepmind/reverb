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

"""Pytype helpers."""

import dataclasses
from typing import Iterable, Mapping, Optional, Union

from reverb import pybind
import tensorflow.compat.v1 as tf

from reverb.cc import schema_pb2
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.saved_model import nested_structure_coder
# pylint: enable=g-direct-tensorflow-import


Fifo = pybind.FifoSelector
Heap = pybind.HeapSelector
Lifo = pybind.LifoSelector
Prioritized = pybind.PrioritizedSelector
Uniform = pybind.UniformSelector

SelectorType = Union[Fifo, Heap, Lifo, Prioritized, Uniform]

# Note that this is effectively treated as `Any`; see b/109648354.
SpecNest = Union[
    tf.TensorSpec, Iterable['SpecNest'], Mapping[str, 'SpecNest']]  # pytype: disable=not-supported-yet


@dataclasses.dataclass
class TableInfo:
  """A tuple describing Table information.

  The main difference between this object and a `schema_pb2.TableInfo` message
  is that the signature is a nested structure of `tf.TypeSpec` objects,
  instead of a raw proto.

  It also has a `TableInfo.from_serialized_proto` classmethod, which is an
  alternate constructor for creating a `TableInfo` object from a serialized
  `schema_pb2.TableInfo` proto.
  """
  # LINT.IfChange
  name: str
  sampler_options: schema_pb2.KeyDistributionOptions
  remover_options: schema_pb2.KeyDistributionOptions
  max_size: int
  max_times_sampled: int
  rate_limiter_info: schema_pb2.RateLimiterInfo
  signature: Optional[SpecNest]
  current_size: int
  num_episodes: int
  num_deleted_episodes: int
  num_unique_samples: int
  table_worker_time: schema_pb2.TableWorkerTime
  # LINT.ThenChange(../../reverb/schema.proto)

  @classmethod
  def from_serialized_proto(cls, proto_string: bytes) -> 'TableInfo':
    """Constructs a TableInfo from a serialized `schema_pb2.TableInfo`."""
    proto = schema_pb2.TableInfo.FromString(proto_string)
    if proto.HasField('signature'):
      signature = nested_structure_coder.decode_proto(proto.signature)
    else:
      signature = None
    return cls(
        name=proto.name,
        sampler_options=proto.sampler_options,
        remover_options=proto.remover_options,
        max_size=proto.max_size,
        max_times_sampled=proto.max_times_sampled,
        rate_limiter_info=proto.rate_limiter_info,
        signature=signature,
        current_size=proto.current_size,
        num_episodes=proto.num_episodes,
        num_deleted_episodes=proto.num_deleted_episodes,
        num_unique_samples=proto.num_unique_samples,
        table_worker_time=proto.table_worker_time,
        )
