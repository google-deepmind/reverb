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

"""Functions that will help construct a reverb.Server from protos.
"""

from typing import Sequence

import reverb
from reverb import reverb_types

from reverb.cc import schema_pb2
from reverb.cc.checkpointing import checkpoint_pb2


def selector_from_proto(
    s: schema_pb2.KeyDistributionOptions
) -> reverb_types.SelectorType:
  """Convert protobuf to reverb_types.SelectorType."""
  if s.fifo:
    return reverb.selectors.Fifo()
  elif s.uniform:
    return reverb.selectors.Uniform()
  elif s.lifo:
    return reverb.selectors.Lifo()
  elif s.WhichOneof('distribution') == 'heap':
    if s.heap.min_heap:
      return reverb.selectors.MinHeap()
    else:
      return reverb.selectors.MaxHeap()
  elif s.WhichOneof('distribution') == 'prioritized':
    return reverb.selectors.Prioritized(
        s.prioritized.priority_exponent)
  else:
    simple_booleans_options = ('fifo', 'lifo', 'uniform')
    if s.WhichOneof('distribution') in simple_booleans_options:
      raise ValueError(f'distribution={s.WhichOneof("distribution")}'
                       ' but the associated boolean value is false.')
    else:
      raise NotImplementedError(
          f'distribution={s.WhichOneof("distribution")}')


def rate_limiter_from_proto(
    proto: checkpoint_pb2.RateLimiterCheckpoint
) -> reverb.rate_limiters.RateLimiter:
  return reverb.rate_limiters.RateLimiter(
      samples_per_insert=proto.samples_per_insert,
      min_size_to_sample=proto.min_size_to_sample,
      min_diff=proto.min_diff,
      max_diff=proto.max_diff)


def tables_from_proto(
    configs: Sequence[checkpoint_pb2.PriorityTableCheckpoint]
) -> Sequence[reverb.Table]:
  """Convert protobuf to reverb.Table."""
  tables = []
  for config in configs:
    tables.append(
        reverb.Table(
            name=config.table_name,
            sampler=selector_from_proto(config.sampler),
            remover=selector_from_proto(config.remover),
            max_size=config.max_size,
            rate_limiter=rate_limiter_from_proto(config.rate_limiter),
            max_times_sampled=config.max_times_sampled,
        ))
  return tables
