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

"""Rate limiters."""

import abc
import sys

from typing import Tuple, Union
from absl import logging
from reverb import pybind


class RateLimiter(metaclass=abc.ABCMeta):
  """Base class for RateLimiters."""

  def __init__(self, samples_per_insert: float, min_size_to_sample: int,
               min_diff: float, max_diff: float):
    self._samples_per_insert = samples_per_insert
    self._min_size_to_sample = min_size_to_sample
    self._min_diff = min_diff
    self._max_diff = max_diff
    self.internal_limiter = pybind.RateLimiter(
        samples_per_insert=samples_per_insert,
        min_size_to_sample=min_size_to_sample,
        min_diff=min_diff,
        max_diff=max_diff)

  def __repr__(self):
    return repr(self.internal_limiter)


class MinSize(RateLimiter):
  """Block sample calls unless replay contains `min_size_to_sample`.

  This limiter blocks all sample calls when the replay contains less than
  `min_size_to_sample` items, and accepts all sample calls otherwise.
  """

  def __init__(self, min_size_to_sample: int):
    if min_size_to_sample < 1:
      raise ValueError(
          f'min_size_to_sample ({min_size_to_sample}) must be a positive '
          f'integer')

    super().__init__(
        samples_per_insert=1.0,
        min_size_to_sample=min_size_to_sample,
        min_diff=-sys.float_info.max,
        max_diff=sys.float_info.max)


class SampleToInsertRatio(RateLimiter):
  """Maintains a specified ratio between samples and inserts.

  The limiter works in two stages:

    Stage 1. Size of table is lt `min_size_to_sample`.
    Stage 2. Size of table is ge `min_size_to_sample`.

  During stage 1 the limiter works exactly like MinSize, i.e. it allows
  all insert calls and blocks all sample calls. Note that it is possible to
  transition into stage 1 from stage 2 when items are removed from the table.

  During stage 2 the limiter attempts to maintain the ratio
  `samples_per_inserts` between the samples and inserts. This is done by
  measuring the "error" in this ratio, calculated as:

    number_of_inserts * samples_per_insert - number_of_samples

  If `error_buffer` is a number and this quantity is larger than
  `min_size_to_sample * samples_per_insert + error_buffer` then insert calls
  will be blocked; sampling will be blocked for error less than
  `min_size_to_sample * samples_per_insert - error_buffer`.

  If `error_buffer` is a tuple of two numbers then insert calls will block if
  the error is larger than error_buffer[1], and sampling will block if the error
  is less than error_buffer[0].

  `error_buffer` exists to avoid unnecessary blocking for a system that is
  more or less in equilibrium.
  """

  def __init__(self, samples_per_insert: float, min_size_to_sample: int,
               error_buffer: Union[float, Tuple[float, float]]):
    """Constructor of SampleToInsertRatio.

    Args:
      samples_per_insert: The average number of times the learner should sample
        each item in the replay error_buffer during the item's entire lifetime.
      min_size_to_sample: The minimum number of items that the table must
        contain  before transitioning into stage 2.
      error_buffer: Maximum size of the "error" before calls should be blocked.
        When a single value is provided then inferred range is
          (
            min_size_to_sample * samples_per_insert - error_buffer,
            min_size_to_sample * samples_per_insert + error_buffer
          )
        The offset is added so that the error tracked is for the insert/sample
        ratio only takes into account operatons occurring AFTER stage 1. If a
        range (two float tuple) then the values are used without any offset.

    Raises:
      ValueError: If error_buffer is smaller than max(1.0, samples_per_inserts).
    """
    if isinstance(error_buffer, float) or isinstance(error_buffer, int):
      offset = samples_per_insert * min_size_to_sample
      min_diff = offset - error_buffer
      max_diff = offset + error_buffer
    else:
      min_diff, max_diff = error_buffer

    if max_diff - min_diff < 2 * max(1.0, samples_per_insert):
      raise ValueError(
          'The size of error_buffer must be >= max(1.0, samples_per_insert) as '
          'smaller values could completely block samples and/or insert calls.'
      )

    if max_diff < samples_per_insert * min_size_to_sample:
      logging.warning(
          'The range covered by error_buffer is below '
          'samples_per_insert * min_size_to_sample. If the sampler cannot '
          'sample concurrently, this will result in a deadlock as soon as '
          'min_size_to_sample items have been inserted.')
    if min_diff > samples_per_insert * min_size_to_sample:
      raise ValueError(
          'The range covered by error_buffer is above '
          'samples_per_insert * min_size_to_sample. This will result in a '
          'deadlock as soon as min_size_to_sample items have been inserted.')

    if min_size_to_sample < 1:
      raise ValueError(
          f'min_size_to_sample ({min_size_to_sample}) must be a positive '
          f'integer')

    super().__init__(
        samples_per_insert=samples_per_insert,
        min_size_to_sample=min_size_to_sample,
        min_diff=min_diff,
        max_diff=max_diff)


class Queue(RateLimiter):
  """Effectively turns the priority table into a queue.

  NOTE: Do not use this RateLimiter directly. Use Table.queue instead.
  NOTE: Must be used in conjunction with a Fifo sampler and remover.
  """

  def __init__(self, size: int):
    """Constructor of Queue (do not use directly).

    Args:
      size: Maximum size of the queue.
    """
    super().__init__(
        samples_per_insert=1.0,
        min_size_to_sample=1,
        min_diff=0.0,
        max_diff=size)


class Stack(RateLimiter):
  """Effectively turns the priority table into a stack.

  NOTE: Do not use this RateLimiter directly. Use Table.stack instead.
  NOTE: Must be used in conjunction with a Lifo sampler and remover.
  """

  def __init__(self, size: int):
    """Constructor of Stack (do not use directly).

    Args:
      size: Maximum size of the stack.
    """
    super().__init__(
        samples_per_insert=1.0,
        min_size_to_sample=1,
        min_diff=0.0,
        max_diff=size)
