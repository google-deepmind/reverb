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

TODO(b/204560248): Expand the documentation.
"""

import copy
import datetime

from typing import Any, Callable, NewType, Optional, Sequence

from reverb import errors
from reverb import pybind
from reverb import reverb_types
import tree

from reverb.cc import patterns_pb2

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.saved_model import nested_structure_coder
# pylint: enable=g-direct-tensorflow-import

# TODO(b/204423296): Expose Python abstractions rather than the raw protos.
Config = patterns_pb2.StructuredWriterConfig

Pattern = tree.Structure[patterns_pb2.PatternNode]
ReferenceStep = NewType('ReferenceStep', Any)
PatternTransform = Callable[[ReferenceStep], Pattern]


class StructuredWriter:
  """StructuredWriter uses static patterns to build and insert trajectories.

  TODO(b/204560248): Expand the documentation.
  """

  def __init__(self, cpp_writer: pybind.StructuredWriter):
    self._writer = cpp_writer
    self._data_structure = None
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
      self._data_structure = tree.map_structure(lambda _: None, data)

    if len(flat_data) != self._flat_data_length:
      raise ValueError(
          f'Flattened data has an unexpected length, got {len(flat_data)} '
          f'but wanted {self._flat_data_length}.')

    try:
      if partial_step:
        self._writer.AppendPartial(flat_data)
      else:
        self._writer.Append(flat_data)
    except ValueError as e:
      parts = str(e).split(' for column ')

      # If the error message doesn't have the expected format then we don't want
      # to change anything.
      if len(parts) != 2:
        raise

      # Use the structure to find the path that corresponds to the flat index.
      col_idx, rest = parts[1].split('. ', 1)
      path = tree.flatten_with_path(self._data_structure)[int(col_idx)][0]

      raise ValueError(
          f'{parts[0]} for column {col_idx} (path={path}). {rest}') from e

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

  @property
  def step_is_open(self) -> bool:
    """True if `partial_step` was set in the most recent `append`."""
    return self._writer.step_is_open


class _RefNode:
  """Helper class to make it easier to build `PatternNode`s."""

  def __init__(self, idx: int):
    self._idx = idx

  def __getitem__(self, key):
    if isinstance(key, int):
      key = slice(key)
    elif not isinstance(key, slice):
      raise ValueError(
          f'Key must be int or slice by got {key} (type {type(key)}).')

    return patterns_pb2.PatternNode(
        flat_source_index=self._idx,
        start=key.start,
        stop=key.stop,
        step=key.step)


def create_reference_step(step_structure: tree.Structure[Any]) -> ReferenceStep:
  """Create a reference structure that can be used to build patterns.

  ```python

  step_structure = {
      'a': None,
      'b': {
          'c': None,
          'd': None,
    }
  }
  ref_step = create_reference_step(step_structure)
  pattern = {
      'last_two_a': ref_step['a'][-2:]
      'second_to_last_c': ref['b']['c'][-2]
      'most_recent_d': ref['b']['d'][-1]
  }

  ```

  Args:
    step_structure: Structure of the data which will be passed to
      `StructuredWriter.append`.

  Returns:
    An object with the same structure as `step_structure` except leaf nodes have
      been replaced with a helper object that builds `patterns_pb2.PatternNode`
      objects when __getitem__ is called.
  """
  return tree.unflatten_as(
      step_structure,
      [_RefNode(x) for x in range(len(tree.flatten(step_structure)))])


def pattern_from_transform(
    step_structure: tree.Structure[Any],
    transform: Callable[[ReferenceStep], Pattern]) -> Pattern:
  """Creates a pattern by invoking a transform from step to output structures.

  ```python

  def my_transform(step):
    return {
        'last_two_a': step['a'][-2:]
        'most_recent_b': tree.map_structure(lambda x: x[-1], step['b']),
    }

  step_structure = {
    'a': None,
    'b': {
      'c': None,
      'd': None,
    }
  }

  pattern = pattern_from_transform(step_structure, my_transform)

  ```

  Args:
    step_structure: Structure of the data which will be passed to
      `StructuredWriter.append`.
    transform: Function that creates the trajectory to be inserted from a
      reference structure.

  Returns:
    A structure with `patterns_pb2.PatternNode` as leaf nodes.
  """
  return transform(create_reference_step(step_structure))


def create_config(pattern: Pattern,
                  table: str,
                  conditions: Sequence[patterns_pb2.Condition] = ()):
  structure = tree.map_structure(lambda _: None, pattern)
  return patterns_pb2.StructuredWriterConfig(
      flat=tree.flatten(pattern),
      pattern_structure=nested_structure_coder.encode_structure(structure),
      table=table,
      priority=1.0,
      conditions=conditions)


def unpack_pattern(config: Config) -> Pattern:
  if not config.HasField('pattern_structure'):
    return config.flat
  structure = nested_structure_coder.decode_proto(config.pattern_structure)
  return tree.unflatten_as(structure, config.flat)


def infer_signature(configs: Sequence[Config],
                    step_spec: reverb_types.SpecNest) -> reverb_types.SpecNest:
  """Infers the table signature from the configs that generate its items.

  Args:
    configs: All the configs used to generate items for the table.
    step_spec: A structured example of the step that will be appended to the
      `StructuredWriter`.

  Returns:
    A nested structure of `TensorSpec` describing the trajectories of the table.

  Raises:
    ValueError: If no configs are provided.
    ValueError: If configs doesn't produce trajectories of identical structure.
    ValueError: If configs targets does not all target the same table.
    ValueError: If configs produce trajectories with incompatible tensors (i.e.
      tensors cannot be concatenated).
  """
  if not configs:
    raise ValueError('At least one config must be provided.')

  if any(c.pattern_structure != configs[0].pattern_structure for c in configs):
    raise ValueError(
        'All configs must have exactly the same pattern_structure.')

  if any(c.table != configs[0].table for c in configs):
    raise ValueError(
        f'All configs must target the same table but provided configs '
        f'included {", ".join(sorted(set(c.table for c in configs)))}.')

  flat_step_spec = tree.flatten(step_spec)

  def _validate_and_convert_to_spec(path, *nodes):
    # Check that all nodes share the same dtype.
    dtypes = [flat_step_spec[node.flat_source_index].dtype for node in nodes]
    if any(dtype != dtypes[0] for dtype in dtypes):
      raise ValueError(
          f'Configs produce trajectories with multiple dtypes at {path}. '
          f'Got {dtypes}.')

    # Create shapes for all nodes.
    shapes = []
    for node in nodes:
      shape = list(flat_step_spec[node.flat_source_index].shape)
      if node.HasField('start'):
        length = (node.stop - node.start) // (node.step or 1)
        shape = [length, *shape]

      shapes.append(tensor_shape.TensorShape(shape))

    # Check that all shapes are either completely identical or at least
    # identical in all dimensions but the first.
    if (any(shape.rank != shapes[0].rank for shape in shapes) or
        (shapes[0].rank > 1 and
         any(shape[1:] != shapes[0][1:] for shape in shapes))):
      raise ValueError(
          f'Configs produce trajectories with incompatible shapes at {path}. '
          f'Got {shapes}.')

    # Merge the shapes into a single shape. If the first dimension varies then
    # we set the leading dimension as undefined.
    if all(shape == shapes[0] for shape in shapes):
      merged_shape = shapes[0]
    else:
      merged_shape = [None, *shapes[0][1:]]

    return tensor_spec.TensorSpec(
        shape=merged_shape,
        dtype=dtypes[0],
        name='/'.join(str(x) for x in path))

  patterns = [unpack_pattern(config) for config in configs]
  return tree.map_structure_with_path(_validate_and_convert_to_spec, *patterns)


class _ConditionBuilder:
  """Helper class to make it easier to build conditions."""

  def __init__(self, incomplete_condition: patterns_pb2.Condition):
    self._incomplete_condition = incomplete_condition

  def __mod__(self, cmp: int) -> '_ConditionBuilder':
    incomplete_condition = copy.deepcopy(self._incomplete_condition)
    incomplete_condition.mod_eq.mod = cmp
    return _ConditionBuilder(incomplete_condition)

  def __eq__(self, cmp: int) -> patterns_pb2.Condition:
    condition = copy.deepcopy(self._incomplete_condition)
    if condition.mod_eq.mod:
      condition.mod_eq.eq = cmp
    else:
      condition.eq = cmp
    return condition

  def __ne__(self, cmp: int) -> patterns_pb2.Condition:
    condition = self == cmp
    condition.inverse = True
    return condition

  def __gt__(self, cmp: int) -> patterns_pb2.Condition:
    return self >= cmp + 1

  def __ge__(self, cmp: int) -> patterns_pb2.Condition:
    condition = copy.deepcopy(self._incomplete_condition)
    condition.ge = cmp
    return condition

  def __lt__(self, cmp: int) -> patterns_pb2.Condition:
    return self <= cmp - 1

  def __le__(self, cmp: int) -> patterns_pb2.Condition:
    condition = self > cmp
    condition.inverse = True
    return condition


class Condition:
  """Building blocks to create conditions from."""

  @staticmethod
  def step_index():
    """(Zero) index of the most recent appended step within the episode."""
    return _ConditionBuilder(patterns_pb2.Condition(step_index=True))

  @staticmethod
  def steps_since_applied():
    """Number of added steps since an item was created for this config."""
    return _ConditionBuilder(patterns_pb2.Condition(steps_since_applied=True))

  @staticmethod
  def is_end_episode():
    """True only when end_episode is called on the writer."""
    return patterns_pb2.Condition(is_end_episode=True, eq=1)

  @staticmethod
  def data(step_structure: tree.Structure[Any]):
    """Value of a scalar integer or bool in the source data."""
    flat = [
        _ConditionBuilder(patterns_pb2.Condition(flat_source_index=i))
        for i in range(len(tree.flatten(step_structure)))
    ]
    return tree.unflatten_as(step_structure, flat)
