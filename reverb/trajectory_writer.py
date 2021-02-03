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

"""Implementation of the EXPERIMENTAL new TrajectoryWriter API.

NOTE! The content of this file should be considered experimental and breaking
changes may occur without notice. Please talk to cassirer@ if you wish to alpha
test these features.
"""

import datetime

from typing import Any, Iterator, Optional, Sequence

from reverb import client as client_lib
from reverb import errors
from reverb import pybind
from reverb import replay_sample

import tree


class TrajectoryWriter:
  """Draft implementation of b/177308010.

  Note: The documentation is minimal as this is just a draft proposal to give
  alpha testers something tangible to play around with.
  """

  def __init__(self, client: client_lib.Client, max_chunk_length: int,
               num_keep_alive_refs: int):
    """Constructor of TrajectoryWriter.

    Note: The client is provided to the constructor as opposed to having the
      client construct the writer object. This is a temporary indirection to
      avoid changes to the public API while we iterate on the design.

    TODO(b/177308010): Move construction to client instead.

    TODO(b/178084425): Allow chunking and reference buffer size to be configured
      at the column level.

    Args:
      client: Reverb client connected to server to write to.
      max_chunk_length: Maximum number of data elements appended to a column
        before its content is automatically finalized as a `ChunkData` (allowing
        pending items which reference the chunks content to be sent to the
        server).
      num_keep_alive_refs: The size of the circular buffer which each column
        maintains for the most recent data appended to it. When a data reference
        popped from the buffer it can no longer be referenced by new items. The
        value `num_keep_alive_refs` can therefore be interpreted as maximum
        number of steps which a trajectory can span.
    """
    self._writer = client._client.NewTrajectoryWriter(max_chunk_length,
                                                      num_keep_alive_refs)
    self._structured_history = None

  def __enter__(self) -> 'TrajectoryWriter':
    return self

  def __exit__(self, *_):
    self.flush()

  def __del__(self):
    self.close()

  @property
  def history(self):
    """References to data, grouped by column and structured like appended data.

    Allows recently added data references to be accesses with list indexing
    semantics. However, instead of returning the raw references, the result is
    wrapped in a TrajectoryColumn object before being returned to the called.

    ```python

    writer = TrajectoryWriter(...)

    # Add three steps worth of data.
    first = writer.append({'a': 1, 'b': 100})
    second = writer.append({'a': 2, 'b': 200})
    third = writer.append({'a': 3, 'b': 300})

    # Create a trajectory using the _ColumnHistory helpers.
    from_history = {
       'all_a': writer.history['a'][:],
       'first_b': writer.history['b'][0],
       'last_b': writer.history['b'][-1],
    }
    writer.create_item(table='name', priority=1.0, trajectory=from_history)

    # Is the same as writing.
    explicit = {
        'all_a': TrajectoryColumn([first['a'], second['a'], third['a']]),
        'first_b': TrajectoryColumn([first['b']]),
        'last_b': TrajectoryColumn([third['b']]),
    }
    writer.create_item(table='name', priority=1.0, trajectory=explicit)

    ```

    Raises:
      RuntimeError: If `append` hasn't been called at least once before.
    """
    if self._structured_history is None:
      raise RuntimeError(
          'history cannot be accessed before `append` is called at least once.')

    return self._structured_history

  def append(self, data: Any):
    """Columnwise append of data leaf nodes to internal buffers.

    If this is the first call then the structure of `data` is extracted and used
    to validate `data` of future `append` calls. The structure of `data` needs
    to remain the same across calls. It is however fine to provide partial data.
    Simply set the field as None.

    Args:
      data: The (possibly nested) structure to make available for new items to
        reference.

    Returns:
      References to the data structured just like provided `data`.
    """
    # Unless it is the first step, check that the structure is the same.
    if self._structured_history is not None:
      tree.assert_same_structure(data, self._structured_history, True)
    else:
      self._structured_history = tree.map_structure(lambda _: _ColumnHistory(),
                                                    data)

    # Flatten the data and pass it to the C++ writer for column wise append. In
    # all columns where data is provided (i.e not None) will return a reference
    # to the data (`pybind.WeakCellRef`) which is used to define trajectories
    # for `create_item`. The columns which did not receive a value (i.e None)
    # will return None.
    flat_data_references = self._writer.Append(tree.flatten(data))

    # Structure the references (and None) in the same way the data was provided.
    structured_data_references = tree.unflatten_as(self._structured_history,
                                                   flat_data_references)

    # Append references to respective columns.
    tree.map_structure(
        lambda column, data_reference: column.append(data_reference),
        self._structured_history, structured_data_references)

    # Return the referenced structured in the same way as `data`.
    return structured_data_references

  def create_item(self, table: str, priority: float, trajectory: Any):
    """Enqueue insertion of an item into `table` referencing `trajectory`.

    Note! This method does NOT BLOCK and therefore is not impacted by the table
    rate limiter. To prevent unwanted runahead, `flush` must be called.

    Before creating an item, `trajectory` is validated.

     * Only contain `TrajectoryColumn` objects.
     * All data references must be alive (i.e not yet expired).
     * Data references within a column must have the same dtype and shape.

    Args:
      table: Name of the table to insert the item into.
      priority: The priority used for determining the sample probability of the
        new item.
      trajectory: A structure of `TrajectoryColumn` objects. The structure is
        flattened before passed to the C++ writer.

    Raises:
      TypeError: If trajectory is invalid.
    """
    flat_trajectory = tree.flatten(trajectory)
    if not all(isinstance(col, TrajectoryColumn) for col in flat_trajectory):
      raise TypeError(
          f'All leaves of `trajectory` must be `TrajectoryColumn` but got '
          f'{trajectory}')

    # Pass the flatten trajectory to the C++ writer where it will be validated
    # and if successful then the item is created and enqued for the background
    # worker to send to the server.
    self._writer.InsertItem(table, priority,
                            [list(column) for column in flat_trajectory])

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
      ValueError: If block_until_num_items < 0.
      DeadlineExceededError: If operation did not complete before the timeout.
    """
    if block_until_num_items < 0:
      raise ValueError(
          f'block_until_num_items must be >= 0, got {block_until_num_items}')

    if timeout_ms is None:
      timeout_ms = -1

    try:
      self._writer.Flush(block_until_num_items, timeout_ms)
    except RuntimeError as e:
      if 'Deadline Exceeded' in str(e) and timeout_ms is not None:
        raise errors.DeadlineExceededError(
            f'ServerInfo call did not complete within provided timeout of '
            f'{datetime.timedelta(milliseconds=timeout_ms)}')
      raise

  def end_episode(self,
                  clear_buffers: bool = True,
                  timeout_ms: Optional[int] = None):
    """Flush all pending items and generate a new episode ID.

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
    self._writer.EndEpisode(clear_buffers, timeout_ms)

    if clear_buffers:
      tree.map_structure(lambda x: x.reset(), self._structured_history)

  def close(self):
    self._writer.Close()


class _ColumnHistory:
  """Utility class for making construction of `TrajectoryColumn`s easy."""

  def __init__(self):
    self._data_references = []

  def append(self, ref: Optional[pybind.WeakCellRef]):
    self._data_references.append(ref)

  def reset(self):
    self._data_references.clear()

  def __len__(self) -> int:
    return len(self._data_references)

  def __iter__(self) -> Iterator[pybind.WeakCellRef]:
    return iter(self._data_references)

  def __getitem__(self, val) -> 'TrajectoryColumn':
    if isinstance(val, int):
      return TrajectoryColumn([self._data_references[val]])
    elif isinstance(val, slice):
      return TrajectoryColumn(self._data_references[val])
    else:
      raise TypeError(
          f'_ColumnHistory indices must be integers, not {type(val)}')


class TrajectoryColumn:
  """Column used for building trajectories referenced by table items."""

  def __init__(self, data_references: Sequence[pybind.WeakCellRef]):
    self._data_references = tuple(data_references)

  def __iter__(self) -> Iterator[pybind.WeakCellRef]:
    return iter(self._data_references)


def sample_trajectory(client: client_lib.Client, table: str,
                      structure: Any) -> replay_sample.ReplaySample:
  """Temporary helper method for sampling a trajectory.

  Note! This function is only intended to make it easier for alpha testers to
  experiment with the new API. It will be removed before this file is made
  public.

  Args:
    client: Client connected to the server to sample from.
    table: Name of the table to sample from.
    structure: Structure to unpack flat data as.

  Returns:
    ReplaySample with trajectory unpacked as `structure` in `data`-field.
  """

  sampler = client._client.NewSampler(table, 1, 1, 1)  # pylint: disable=protected-access
  sample = sampler.GetNextSample()
  return replay_sample.ReplaySample(
      info=replay_sample.SampleInfo(
          key=int(sample[0][0]),
          probability=float(sample[1][0]),
          table_size=int(sample[2][0]),
          priority=float(sample[3][0])),
      data=tree.unflatten_as(structure, sample[4:]))
