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

"""Implementation of the TrajectoryWriter."""

import datetime
import itertools
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from reverb import errors
from reverb import pybind
import tree


class TrajectoryWriter:
  """The TrajectoryWriter is used to write data to tables at a Reverb server.

  At a high level, the process of inserting trajectories can be summarized as:

    * Structured data is appended to an internal buffer using the `append`
      method and the caller receives a reference to each element (i.e leaf node)
      in the original data.
    * Compatible data referenced (i.e same dtype and compatible shapes) are
      concatenated into `TrajectoryColumn`s which in turn are combined into a
      trajectory and inserted into a table using the `create_item` method.

  It is important to understand that the structure of the data provided to
  `append` does NOT need to match the structure of the trajectory which the
  sampler will receive when it samples the item. To illustrate, consider a
  scenario were want to sample SARS (State-action-reward-state) style trajectory
  of length 5. Furthermore, we would like a trajectory to start at every step.
  It would look something like this:

  ```python

  client = Client(...)
  env = ....  # Construct the environment
  policy = ....  # Construct the agent's policy

  with client.trajectory_writer(num_keep_alive_refs=5) as writer:
    for episode in range(NUM_EPISODES):
      timestep = env.reset()

      # You probably have strong opinions of whether the actions should be
      # aligned with the step it originated from or the destination. In this
      # example we'll align it with the destination state and thus we'll start
      # off by appending the timestep WITHOUT an action.
      writer.append({
          'observation': timestep.observation,
      })

      while not timestep.last():
        # Select the action according to your policy and act it out in the
        # environment.
        action = policy(timestep)
        timestep = env.step(action)

        # Now we have both an action and the state it resulted in. We append
        # both of these together to the writer. This will result in them
        # sharing the same index in `writer.history`.
        writer.append({
            'observation': timestep.observation,
            'reward': timestep.reward,
            'action': action,
        })

        # Once we have seen at least 5 timesteps (including the first one
        # which does not have an aligned action) then we can start inserting
        # items that reference the last 5 timesteps and the last 4 actions.
        if writer.episode_steps >= 5:
          trajectory = {
              'states': writer.history['observation'][-5:],
              'rewards': writer.history['reward'][-4:],
              'actions': writer.history['action'][-4:],
          }
          writer.create_item(
              table='my_table',
              priority=calc_priority(trajectory),
              trajectory=trajectory)

      # Block until all pending items have been sent to the server and
      # inserted into 'my_table'. This also clears the buffers so history will
      # once again be empty and `writer.episode_steps` is 0.
      writer.end_episode()

  ```
  """

  def __init__(self, internal_writer: pybind.TrajectoryWriter):
    """Constructor of TrajectoryWriter (must only be called by `Client`)."""
    self._writer = internal_writer

    # The union of the structures of all data passed to `append`. The structure
    # grows everytime the provided data contains one or more fields which were
    # not present in any of the data seen before.
    self._structure = None

    # References to all data seen since the writer was constructed or last reset
    # (through end_episode). The number of columns always matches the number of
    # leaf nodes in `_structure` but the order is not (necessarily) the same as
    # `tree.flatten(_structure)` since the structure may evolve over time.
    # Instead the mapping is controlled by `_path_to_column_index`. See
    # `_flatten` and `_unflatten` for more details.
    self._column_history: List[_ColumnHistory] = []

    # Mapping from structured paths (i.e as received from
    # `tree.flatten_with_path`) to position in `_column_history`. This is used
    # in `_flatten`.
    self._path_to_column_index: Mapping[str, int] = {}

    # The inverse of `_path_to_column_index`. That is the mapping describes the
    # swaps required to go from the order of `column_history` (and the C++
    # writer) to the order of a sequence which can be unflattened into
    # `_structure`. This is used in `_unflatten`.
    self._column_index_to_flat_structure_index: Mapping[int, int] = {}
    self._path_to_column_config = {}

    # Set when `append` called with `partial_step=True`. Remains set until
    # `append` called with `partial_step=False`. This is used to control where
    # new data references are added to the history (i.e whether a new step
    # should be created).
    self._last_step_is_open = False

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
    if not self._column_history:
      raise RuntimeError(
          'history cannot be accessed before `append` is called at least once.')

    return self._unflatten(self._column_history)

  @property
  def episode_steps(self) -> int:
    """Number of append calls since last `end_episode` call.

    This does not count partial calls to append, i.e. ones with
    `partial_step=True`.
    """
    if not self._column_history:
      return 0
    else:
      return len(self._column_history[0]) - int(self._last_step_is_open)

  def configure(self, path: Tuple[Union[int, str], ...],
                *,
                num_keep_alive_refs: int,
                max_chunk_length: Optional[int]):
    """Override chunking options for a single column.

    Args:
      path: Structured path to the column to configure.
      num_keep_alive_refs: Override value for `num_keep_alive_refs` i.e the size
        of the circular buffer of the most recently added data.
      max_chunk_length: Override value for the chunk length used by this column.
        When set to None, an auto tuned chunk length is used. When set to a
        number, a constant chunk length is used.

    Raises:
      ValueError: If num_keep_alive_refs is < 1.
      ValueError: If max_chunk_length set to a value < 1 or to a value > than
        num_keep_alive_refs.
    """
    if num_keep_alive_refs < 1:
      raise ValueError(
          f'num_keep_alive_refs ({num_keep_alive_refs}) must be a positive '
          f'integer')
    if max_chunk_length is not None and (
        max_chunk_length < 1 or max_chunk_length > num_keep_alive_refs):
      raise ValueError(
          f'max_chunk_length ({max_chunk_length}) must be None or a positive '
          f'integer <= num_keep_alive_refs ({num_keep_alive_refs})')

    if max_chunk_length is None:
      chunker_options = pybind.AutoTunedChunkerOptions(
          num_keep_alive_refs=num_keep_alive_refs,
          throughput_weight=1.0)
    else:
      chunker_options = pybind.ConstantChunkerOptions(
          max_chunk_length=max_chunk_length,
          num_keep_alive_refs=num_keep_alive_refs)

    if path in self._path_to_column_index:
      self._writer.ConfigureChunker(self._path_to_column_index[path],
                                    chunker_options)
    else:
      self._path_to_column_config[path] = chunker_options

  def append(self, data: Any, *, partial_step: bool = False):
    """Columnwise append of data leaf nodes to internal buffers.

    If `data` includes fields or sub structures which haven't been present in
    any previous calls then the types and shapes of the new fields are extracted
    and used to validate future `append` calls. The structure of `history` is
    also updated to include the union of the structure across all `append`
    calls.

    When new fields are added after the first step then the newly created
    history field will be filled with `None` in all preceding positions. This
    results in the equal indexing across columns. That is `a[i]` and `b[i]`
    references the same step in the sequence even if `b` was first observed
    after `a` had already been seen.

    It is possible to create a "step" using more than one `append` call by
    setting the `partial_step` flag. Partial steps can be used when some parts
    of the step becomes available only as a result of inserting (and learning
    from) trajectories that include the fields available first (e.g learn from
    the SARS trajectory to select the next action in an on-policy agent). In the
    final `append` call of the step, `partial_step` must be set to False.
    Failing to "close" the partial step will result in error as the same field
    must NOT be provided more than once in the same step.

    Args:
      data: The (possibly nested) structure to make available for new items to
        reference.
      partial_step: If `True` then the step is not considered "done" with this
        call. See above for more details. Defaults to `False`.

    Returns:
      References to the data structured just like provided `data`.

    Raises:
      ValueError: If the same column is provided more than once in the same
        step.
    """
    # Unless it is the first step, check that the structure is the same.
    if self._structure is None:
      self._update_structure(tree.map_structure(lambda _: None, data))

    try:
      tree.assert_same_structure(data, self._structure, True)
      expanded_data = data
    except ValueError:
      try:
        # If `data` is a subset of the full spec then we can simply fill in the
        # gaps with None.
        expanded_data = _tree_merge_into(source=data, target=self._structure)
      except ValueError:
        # `data` contains fields which haven't been observed before so we need
        # expand the spec using the union of the history and `data`.
        self._update_structure(
            _tree_union(self._structure,
                        tree.map_structure(lambda x: None, data)))

        # Now that the structure has been updated to include all the fields in
        # `data` we are able to expand `data` to the full structure. Note that
        # if `data` is a superset of the previous history structure then this
        # "expansion" is just a noop.
        expanded_data = _tree_merge_into(data, self._structure)

    # Use our custom mapping to flatten the expanded structure into columns.
    flat_column_data = self._flatten(expanded_data)

    # If the last step is still open then verify that already populated columns
    # are None in the new `data`.
    if self._last_step_is_open:
      for i, (column, column_data) in enumerate(
          zip(self._column_history, flat_column_data)):
        if column_data is None or column.can_set_last:
          continue

        raise ValueError(
            f'Field {self._get_path_for_column_index(i)} has already been set '
            f'in the active step by previous (partial) append call and thus '
            f'must be omitted or set to None but got: {column_data}')

    # Flatten the data and pass it to the C++ writer for column wise append. In
    # all columns where data is provided (i.e not None) will return a reference
    # to the data (`pybind.WeakCellRef`) which is used to define trajectories
    # for `create_item`. The columns which did not receive a value (i.e None)
    # will return None.
    if partial_step:
      flat_column_data_references = self._writer.AppendPartial(flat_column_data)
    else:
      flat_column_data_references = self._writer.Append(flat_column_data)

    # Append references to respective columns. Note that we use the expanded
    # structure in order to populate the columns missing from the data with
    # None.
    for column, data_reference in zip(self._column_history,
                                      flat_column_data_references):
      # If the last step is still open (i.e `partial_step` was set) then we
      # populate that step instead of creating a new one.
      if not self._last_step_is_open:
        column.append(data_reference)
      elif data_reference is not None:
        column.set_last(data_reference)

    # Save the flag so the next `append` call either populates the same step
    # or begins a new step.
    self._last_step_is_open = partial_step

    # Unpack the column data into the expanded structure.
    expanded_structured_data_references = self._unflatten(
        flat_column_data_references)

    # Return the referenced structured in the same way as `data`. If only a
    # subset of the fields were present in the input data then only these fields
    # will exist in the output.
    return _tree_filter(expanded_structured_data_references, data)

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
    self._writer.CreateItem(table, priority,
                            [list(column) for column in flat_trajectory],
                            [column.is_squeezed for column in flat_trajectory])

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
      if 'Timeout exceeded' in str(e) and timeout_ms is not None:
        raise errors.DeadlineExceededError(
            f'Flush call did not complete within provided timeout of '
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
      for column in self._column_history:
        column.reset()

  def close(self):
    self._writer.Close()

  def _flatten(self, data):
    flat_data = [None] * len(self._path_to_column_index)
    for path, value in tree.flatten_with_path(data):
      flat_data[self._path_to_column_index[path]] = value
    return flat_data

  def _unflatten(self, flat_data):
    reordered_flat_data = [
        flat_data[self._column_index_to_flat_structure_index[i]]
        for i in range(len(flat_data))
    ]
    return tree.unflatten_as(self._structure, reordered_flat_data)

  def _get_path_for_column_index(self, column_index):
    i = self._column_index_to_flat_structure_index[column_index]
    return tree.flatten_with_path(self._structure)[i][0]

  def _update_structure(self, new_structure: Any):
    """Replace the existing structure with a superset of the current one.

    Since the structure is allowed to evolve over time we are unable to simply
    map flattened data to column indices. For example, if the first step is
    `{'a': 1, 'c': 101}` and the second step is `{'a': 2, 'b': 12, 'c': 102}`
    then the flatten data would be `[1, 101]` and `[2, 12, 102]`. This will
    result in invalid behaviour as the second column (index 1) would receive `c`
    in the first step and `b` in the second.

    To mitigate this we maintain an explicit mapping from path -> column. The
    mapping is allowed to grow over time and would in the above example be
    `{'a': 0, 'c': 1}` and `{'a': 0, 'b': 2, 'c': 1}` after the first and second
    step resp. Data would thus be flatten as `[1, 101]` and `[2, 102, 12]` which
    means that the columns in the C++ layer only receive data from a single
    field in the structure even if it evolves over time.

    Args:
      new_structure: The new structure to use. Must be a superset of the
        previous structure.
    """
    new_structure_with_path_flat = tree.flatten_with_path(new_structure)
    # Evolve the mapping from structure path to column index.
    for path, _ in new_structure_with_path_flat:
      if path not in self._path_to_column_index:
        self._path_to_column_index[path] = len(self._path_to_column_index)

        # If an explicit config have been provided for the column then forward
        # it to the C++ writer so it will be applied when the column chunker is
        # created.
        if path in self._path_to_column_config:
          self._writer.ConfigureChunker(self._path_to_column_index[path],
                                        self._path_to_column_config[path])

    # Recalculate the reverse mapping, i.e column index to index within the
    # flatten structure.
    self._column_index_to_flat_structure_index = {
        i: self._path_to_column_index[path]
        for i, (path, _) in enumerate(new_structure_with_path_flat)
    }

    # New columns are always added to the back so all we need to do expand the
    # history structure is to append one column for every field added by this
    # `_update_structure` call.  In order to align indexing across all columns
    # we init the new fields with None for all steps up until this.
    history_length = len(self._column_history[0]) if self._column_history else 0
    while len(self._column_history) < len(tree.flatten(new_structure)):
      column_index = len(self._column_history)
      self._column_history.append(
          _ColumnHistory(new_structure_with_path_flat[column_index][0],
                         history_length))

    # With the mapping and history updated the structure can be set.
    self._structure = new_structure


class _ColumnHistory:
  """Utility class for building `TrajectoryColumn`s from structured history."""

  def __init__(self,
               path: Tuple[Union[str, int], ...],
               history_padding: int = 0):
    """Constructor for _ColumnHistory.

    Args:
      path: A Tuple of strings and ints that represents which leaf-node this
        column represents in TrajectoryWriter._structure.
      history_padding: The number of Nones used to forward-pad the column's
        history.
    """
    self._path = path
    self._data_references: Sequence[Optional[pybind.WeakCellRef]] = (
        [None] * history_padding)

  def append(self, ref: Optional[pybind.WeakCellRef]):
    self._data_references.append(ref)

  def reset(self):
    self._data_references.clear()

  def set_last(self, ref: pybind.WeakCellRef):
    if not self._data_references:
      raise RuntimeError('set_last called on empty history column')
    if self._data_references[-1] is not None:
      raise RuntimeError('set_last called on already set cell')
    self._data_references[-1] = ref

  @property
  def can_set_last(self) -> bool:
    return self._data_references and self._data_references[-1] is None

  def __len__(self) -> int:
    return len(self._data_references)

  def __iter__(self) -> Iterator[Optional[pybind.WeakCellRef]]:
    return iter(self._data_references)

  def __getitem__(self, val) -> 'TrajectoryColumn':
    path = self._path + (val,)
    if isinstance(val, int):
      return TrajectoryColumn([self._data_references[val]],
                              squeeze=True,
                              path=path)
    elif isinstance(val, slice):
      return TrajectoryColumn(
          self._data_references[val], path=path)
    else:
      raise TypeError(
          f'_ColumnHistory indices must be integers, not {type(val)}')

  def __str__(self):
    name = f'{self.__class__.__module__}.{self.__class__.__name__}'
    return f'{name}(path={self._path}, refs={self._data_references})'


class TrajectoryColumn:
  """Column used for building trajectories referenced by table items."""

  def __init__(self,
               data_references: Sequence[pybind.WeakCellRef],
               *,
               squeeze: bool = False,
               path: Tuple[Union[str, int, slice], ...] = None):
    if squeeze and len(data_references) != 1:
      raise ValueError(
          f'Columns must contain exactly one data reference when squeeze set, '
          f'got {len(data_references)}')
    if any(ref is None for ref in data_references):
      raise ValueError('TrajectoryColumns cannot contain any None data '
                       f'references, got {data_references} for '
                       f'TrajectoryColumn at path {path}')

    self._data_references = tuple(data_references)
    self.is_squeezed = squeeze

  def __len__(self) -> int:
    return len(self._data_references)

  def __iter__(self) -> Iterator[pybind.WeakCellRef]:
    return iter(self._data_references)

  def __getitem__(self, val) -> 'TrajectoryColumn':
    if isinstance(val, int):
      return TrajectoryColumn([self._data_references[val]], squeeze=True)
    elif isinstance(val, slice):
      return TrajectoryColumn(self._data_references[val])
    else:
      raise TypeError(
          f'TrajectoryColumn indices must be integers or slices, '
          f'not {type(val)}')

  @property
  def shape(self) -> Tuple[Optional[int], ...]:
    return (len(self._data_references), *self._data_references[0].shape)

  @property
  def dtype(self) -> np.dtype:
    return self._data_references[0].dtype

  def numpy(self) -> np.ndarray:
    """Gets and stacks all the referenced data.

    Data is copied from buffers in the C++ layers and may involve decompression
    of already created chunks. This can be quite a memory intensive operation
    when used on large arrays.

    Returns:
      All referenced data stacked in a single numpy array if column isn't
      squeezed. If the column is squeezed then the value is returned without
      stacking.

    Raises:
      RuntimeError: If any data reference has expired.
    """
    if any(reference.expired for reference in self._data_references):
      raise RuntimeError(
          'Cannot convert TrajectoryColumn with expired data references to '
          'numpy array.')

    if self.is_squeezed:
      return self._data_references[0].numpy()

    return np.stack([ref.numpy() for ref in self._data_references])


def _tree_merge_into(source, target):
  """Update `target` with content of substructure `source`."""
  path_to_index = {
      path: i for i, (path, _) in enumerate(tree.flatten_with_path(target))
  }

  flat_target = tree.flatten(target)
  for path, leaf in tree.flatten_with_path(source):
    if path not in path_to_index:
      raise ValueError(
          f'Cannot expand {source} into {target} as it is not a sub structure.')
    flat_target[path_to_index[path]] = leaf

  return tree.unflatten_as(target, flat_target)


def _tree_filter(source, filter_):
  """Extract `filter_` from `source`."""
  path_to_index = {
      path: i for i, (path, _) in enumerate(tree.flatten_with_path(filter_))
  }

  flat_target = [None] * len(path_to_index)
  for path, leaf in tree.flatten_with_path(source):
    if path in path_to_index:
      flat_target[path_to_index[path]] = leaf

  return tree.unflatten_as(filter_, flat_target)


def _is_named_tuple(x):
  # Classes that look syntactically as if they inherit from `NamedTuple` in
  # fact end up not doing so, so use this heuristic to detect them.
  return isinstance(x, Tuple) and hasattr(x, '_fields')


def _tree_union(a, b):
  """Compute the disjunction of two trees with None leaves."""
  if a is None:
    return a

  if _is_named_tuple(a):
    return type(a)(**_tree_union(a._asdict(), b._asdict()))
  if isinstance(a, (List, Tuple)):
    return type(a)(
        _tree_union(aa, bb) for aa, bb in itertools.zip_longest(a, b))

  merged = {**a}

  for k, v in b.items():
    if k in a:
      merged[k] = _tree_union(a[k], v)
    else:
      merged[k] = v

  return type(a)(**merged)
