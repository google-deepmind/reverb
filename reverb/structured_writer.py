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

"""StructuredWriter uses static patterns to build and insert trajectories.

NOTE! This class is EXPERIMENTAL and not yet ready for use.

TODO(b/204421540): Remove warning once it is ready for use.

TODO(b/204560248): Expand the documentation.
"""

import datetime

from typing import Any, Optional

from reverb import errors
from reverb import pybind
import tree

from reverb.cc import patterns_pb2

# TODO(b/204423296): Expose Python abstractions rather than the raw protos.
Config = patterns_pb2.StructuredWriterConfig
Node = patterns_pb2.PatternNode
Condition = patterns_pb2.Condition
ModuloEq = patterns_pb2.Condition.ModuloEq


class StructuredWriter:
  """StructuredWriter uses static patterns to build and insert trajectories.

  NOTE! This class is EXPERIMENTAL and not yet ready for use.

  TODO(b/204421540): Remove warning once it is ready for use.

  TODO(b/204560248): Expand the documentation.
  """

  def __init__(self, cpp_writer: pybind.StructuredWriter):
    self._writer = cpp_writer
    self._flat_data_length = None

  def append(self, data: Any, *, partial_step: bool = False):
    """Appends data to internal buffers and inserts generated trajectories.

    NOTE! The data must have exactly the same structure in each step. Leaf nodes
    are allowed to be `None` but the structure must be the same.

    It is possible to create a "step" using more than one `append` call by
    setting the `partial_step` flag. Partial steps can be used when some parts
    of the step becomes available only as a result of inserting (and learning
    from) trajectories that include the fields available first (e.g learn from
    the SARS trajectory to select the next action in an on-policy agent). In the
    final `append` call of the step, `partial_step` must be set to `False`.
    Failing to "close" the partial step will result in error as the same field
    must NOT be provided more than once in the same step.

    Args:
      data: The (possibly nested) data pushed to internal buffers.
      partial_step: If `True` then the step is not considered "done" with this
        call. See above for more details. Defaults to `False`.

    Raises:
      ValueError: If the number of items in the flattened data changes between
        calls.
    """
    flat_data = tree.flatten(data)

    if self._flat_data_length is None:
      self._flat_data_length = len(flat_data)

    if len(flat_data) != self._flat_data_length:
      raise ValueError(
          f'Flattened data has an unexpected length, got {len(flat_data)} '
          f' but wanted {self._flat_data_length}')

    if partial_step:
      self._writer.AppendPartial(flat_data)
    else:
      self._writer.Append(flat_data)

  def flush(self,
            block_until_num_items: int = 0,
            timeout_ms: Optional[int] = None):
    """Block until all but `block_until_num_items` confirmed by the server.

    There are two ways that an item could be "pending":

      1. Some of the data elements referenced by the item have not yet been
         finalized (and compressed) as a `ChunkData`.
      2. The item has been written to the gRPC stream but the response
         confirming the insertion has not yet been received.

    Type 1 pending items are transformed into type 2 when flush is called by
    forcing (premature) chunk finalization of the data elements referenced by
    the items. This will allow the background worker to write the data and items
    to the gRPC stream and turn them into type 2 pending items.

    The time it takes for type 2 pending items to be confirmed is primarily
    due to the state of the table rate limiter. After the items have been
    written to the gRPC stream then all we can do is wait (GIL is not held).

    Args:
      block_until_num_items: If > 0 then this many pending items will be allowed
        to remain as type 1. If the number of type 1 pending items is less than
        `block_until_num_items` then we simply wait until the total number of
        pending items is <= `block_until_num_items`.
      timeout_ms: (optional, default is no timeout) Maximum time to block for
        before unblocking and raising a `DeadlineExceededError` instead. Note
        that although the block is interrupted, the insertion of the items will
        proceed in the background.

    Raises:
      ValueError: If `block_until_num_items` < 0.
      DeadlineExceededError: If operation did not complete before the timeout.
    """
    if block_until_num_items < 0:
      raise ValueError(
          f'block_until_num_items must be >= 0, got {block_until_num_items}')

    try:
      self._writer.Flush(block_until_num_items, timeout_ms)
    except RuntimeError as e:
      if 'Timeout exceeded' in str(e) and timeout_ms is not None:
        raise errors.DeadlineExceededError(
            f'Flush call did not complete within provided timeout of '
            f'{datetime.timedelta(milliseconds=timeout_ms)}')
      raise

  def end_episode(self,
                  clear_buffers: bool = True,
                  timeout_ms: Optional[int] = None):
    """Flush all pending items and generate a new episode ID.

    Configurations that are conditioned to only be appied on episode end are
    applied (assuming all other conditions are fulfilled) and the items inserted
    before flush is called.

    Args:
      clear_buffers: Whether the history should be cleared or not. Buffers
        should only not be cleared when trajectories spanning multiple episodes
        are used.
      timeout_ms: (optional, default is no timeout) Maximum time to block for
        before unblocking and raising a `DeadlineExceededError` instead. Note
        that although the block is interrupted, the buffers and episode ID are
        reset all the same and the insertion of the items will proceed in the
        background thread.

    Raises:
      DeadlineExceededError: If operation did not complete before the timeout.
    """
    try:
      self._writer.EndEpisode(clear_buffers, timeout_ms)
    except RuntimeError as e:
      if 'Timeout exceeded' in str(e) and timeout_ms is not None:
        raise errors.DeadlineExceededError(
            f'End episode call did not complete within provided timeout of '
            f'{datetime.timedelta(milliseconds=timeout_ms)}')
      raise
