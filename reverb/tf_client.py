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

"""TFClient provides tf-ops for interacting with Reverb."""

from typing import Any, List, Optional, Sequence

from reverb import replay_sample
import tensorflow.compat.v1 as tf
import tree

from reverb.cc.ops import gen_client_ops
from reverb.cc.ops import gen_dataset_op


class ReplayDataset(tf.data.Dataset):
  """A tf.data.Dataset which samples timesteps from the ReplayService.

  Note: The dataset returns `ReplaySample` where `data` with the structure of
  `dtypes` and `shapes`.

  Note: Uses of Python lists are converted into tuples as nest used by the
  tf.data API doesn't have good support for lists.

  Timesteps are streamed through the dataset as follows:

    1. Does an active prioritized item exists?
         - Yes: Go to 3
         - No: Go to 2.
    2. Sample a prioritized item from `table` using its sample-function and set
       the item as "active". Go to 3.
    3. Yield the next timestep within the active prioritized item. If the
       timestep was the last one within the item, clear its "active" status.

  This allows for items of arbitrary length to be streamed with limited memory.
  """

  def __init__(self,
               server_address: str,
               table: str,
               dtypes: Any,
               shapes: Any,
               max_in_flight_samples_per_worker: int,
               num_workers_per_iterator: int = -1,
               max_samples_per_stream: int = -1,
               sequence_length: Optional[int] = None,
               emit_timesteps: bool = True):
    """Constructs a new ReplayDataset.

    Args:
      server_address: Address of gRPC ReplayService.
      table: Probability table to sample from.
      dtypes: Dtypes of the data output. Can be nested.
      shapes: Shapes of the data output. Can be nested.
      max_in_flight_samples_per_worker: The number of samples requested in each
        batch of samples. Higher values give higher throughput but too big
        values can result in skewed sampling distributions as large number of
        samples are fetched from single snapshot of the replay (followed by a
        period of lower activity as the samples are consumed). A good rule of
        thumb is to set this value to 2-3x times the batch size used.
      num_workers_per_iterator: (Defaults to -1, i.e auto selected) The
        number of worker threads to create per dataset iterator. When the
        selected table uses a FIFO sampler (i.e a queue) then exactly 1 worker
        must be used to avoid races causing invalid ordering of items. For all
        other samplers, this value should be roughly equal to the number of
        threads available on the CPU.
      max_samples_per_stream: (Defaults to -1, i.e auto selected) The
        maximum number of samples to fetch from a stream before a new call is
        made. Keeping this number low ensures that the data is fetched
        uniformly from all server.
      sequence_length: (Defaults to None, i.e unknown) The number of timesteps
        that each sample consists of. If set then the length of samples received
        from the server will be validated against this number.
      emit_timesteps: (Defaults to True) If set, timesteps instead of full
        sequences are returned from the dataset. Returning sequences instead
        of timesteps can be more efficient as the memcopies caused by the
        splitting and batching of tensor can be avoided. Note that if set to
        False then then all `shapes` must have dim[0] equal to
        `sequence_length`.

    Raises:
      ValueError: If `dtypes` and `shapes` don't share the same structure.
      ValueError: If max_in_flight_samples_per_worker is not a positive integer.
      ValueError: If num_workers_per_iterator is not a positive integer or -1.
      ValueError: If max_samples_per_stream is not a positive integer or -1.
      ValueError: If sequence_length is not a positive integer or None.
      ValueError: If emit_timesteps is False and not all items in shapes has
        sequence_length as its leading dimension.
    """
    tree.assert_same_structure(dtypes, shapes, False)
    if max_in_flight_samples_per_worker < 1:
      raise ValueError(
          'max_in_flight_samples_per_worker (%d) must be a positive integer' %
          max_in_flight_samples_per_worker)
    if num_workers_per_iterator < 1 and num_workers_per_iterator != -1:
      raise ValueError(
          'num_workers_per_iterator (%d) must be a positive integer or -1' %
          num_workers_per_iterator)
    if max_samples_per_stream < 1 and max_samples_per_stream != -1:
      raise ValueError(
          'max_samples_per_stream (%d) must be a positive integer or -1' %
          max_samples_per_stream)
    if sequence_length is not None and sequence_length < 1:
      raise ValueError(
          'sequence_length (%s) must be None or a positive integer' %
          sequence_length)

    # Add the info fields.
    dtypes = replay_sample.ReplaySample(replay_sample.SampleInfo.tf_dtypes(),
                                        dtypes)
    shapes = replay_sample.ReplaySample(
        replay_sample.SampleInfo(
            tf.TensorShape([sequence_length] if not emit_timesteps else []),
            tf.TensorShape([sequence_length] if not emit_timesteps else []),
            tf.TensorShape([sequence_length] if not emit_timesteps else [])),
        shapes)

    # If sequences are to be emitted then all shapes must specify use
    # sequence_length as their batch dimension.
    if not emit_timesteps:

      def _validate_batch_dim(path: str, shape: tf.TensorShape):
        if (not shape.ndims
            or tf.compat.dimension_value(shape[0]) != sequence_length):
          raise ValueError(
              'All items in shapes must use sequence_range (%s) as the leading '
              'dimension, but "%s" has shape %s' %
              (sequence_length, path[0], shape))

      tree.map_structure_with_path(_validate_batch_dim, shapes.data)

    # The tf.data API doesn't fully support lists so we convert all uses of
    # lists into tuples.
    dtypes = _convert_lists_to_tuples(dtypes)
    shapes = _convert_lists_to_tuples(shapes)

    self._server_address = server_address
    self._table = table
    self._dtypes = dtypes
    self._shapes = shapes
    self._sequence_length = sequence_length
    self._emit_timesteps = emit_timesteps
    self._max_in_flight_samples_per_worker = max_in_flight_samples_per_worker
    self._num_workers_per_iterator = num_workers_per_iterator
    self._max_samples_per_stream = max_samples_per_stream

    if _is_tf1_runtime():
      # Disabling to avoid errors given the different tf.data.Dataset init args
      # between v1 and v2 APIs.
      # pytype: disable=wrong-arg-count
      super().__init__()
    else:
      # DatasetV2 requires the dataset as a variant tensor during init.
      super().__init__(self._as_variant_tensor())
      # pytype: enable=wrong-arg-count

  def _as_variant_tensor(self):
    return gen_dataset_op.reverb_dataset(
        server_address=self._server_address,
        table=self._table,
        dtypes=tree.flatten(self._dtypes),
        shapes=tree.flatten(self._shapes),
        emit_timesteps=self._emit_timesteps,
        sequence_length=self._sequence_length or -1,
        max_in_flight_samples_per_worker=self._max_in_flight_samples_per_worker,
        num_workers_per_iterator=self._num_workers_per_iterator,
        max_samples_per_stream=self._max_samples_per_stream)

  def _inputs(self) -> List[Any]:
    return []

  @property
  def element_spec(self) -> Any:
    return tree.map_structure(tf.TensorSpec, self._shapes, self._dtypes)


class TFClient:
  """Client class for calling Reverb replay servers from a TensorFlow graph."""

  def __init__(self,
               server_address: str,
               shared_name: Optional[str] = None,
               name: str = 'reverb'):
    """Creates the client TensorFlow handle.

    Args:
      server_address: Address of the server.
      shared_name: (Optional) If non-empty, this client will be shared under the
        given name across multiple sessions.
      name: Optional name for the Client operations.
    """
    self._name = name
    self._server_address = server_address
    self._handle = gen_client_ops.reverb_client(
        server_address=server_address, shared_name=shared_name, name=name)

  def sample(self,
             table: str,
             data_dtypes,
             name: Optional[str] = None) -> replay_sample.ReplaySample:
    """Samples an item from the replay.

    This only allows sampling items with a data field.

    Args:
      table: Probability table to sample from.
      data_dtypes: Dtypes of the data output. Can be nested.
      name: Optional name for the Client operations.

    Returns:
      A ReplaySample with data nested according to data_dtypes. See ReplaySample
      for more details.
    """
    with tf.name_scope(name, f'{self._name}_sample', ['sample']) as scope:
      key, probability, table_size, data = gen_client_ops.reverb_client_sample(
          self._handle, table, tree.flatten(data_dtypes), name=scope)
      return replay_sample.ReplaySample(
          replay_sample.SampleInfo(key, probability, table_size),
          tree.unflatten_as(data_dtypes, data))

  def insert(self,
             data: Sequence[tf.Tensor],
             tables: tf.Tensor,
             priorities: tf.Tensor,
             name: Optional[str] = None):
    """Inserts a trajectory into one or more tables.

    The content of `tables` and `priorities` are zipped to create the
    prioritized items. That is, an item with priority `priorities[i]` is
    inserted into `tables[i]`.

    Args:
      data: Tensors to insert as the trajectory.
      tables: Rank 1 tensor with the names of the tables to create prioritized
        items in.
      priorities: Rank 1 tensor with priorities of the new items.
      name: Optional name for the client operation.

    Returns:
      A tf-op for performing the insert.

    Raises:
      ValueError: If tables is not a string tensor of rank 1.
      ValueError: If priorities is not a float64 tensor of rank 1.
      ValueError: If priorities and tables does not have the same shape.
    """
    if tables.dtype != tf.string or tables.shape.rank != 1:
      raise ValueError('tables must be a string tensor of rank 1')
    if priorities.dtype != tf.float64 or priorities.shape.rank != 1:
      raise ValueError('priorities must be a float64 tensor of rank 1')
    if not tables.shape.is_compatible_with(priorities.shape):
      raise ValueError('priorities and tables must have the same shape')

    with tf.name_scope(name, f'{self._name}_insert', ['insert']) as scope:
      return gen_client_ops.reverb_client_insert(
          self._handle, data, tables, priorities, name=scope)

  def update_priorities(self,
                        table: str,
                        keys: tf.Tensor,
                        priorities: tf.Tensor,
                        name: str = None):
    """Creates op for updating priorities of existing items in the replay.

    Not found elements for `keys` are silently ignored.

    Args:
      table: Probability table to update.
      keys: Keys of the items to update. Must be same length as `priorities`.
      priorities: New priorities for `keys`. Must be same length as `keys`.
      name: Optional name for the operation.

    Returns:
      A tf-op for performing the update.
    """

    with tf.name_scope(name, f'{self._name}_update_priorities',
                       ['update_priorities']) as scope:
      return gen_client_ops.reverb_client_update_priorities(
          self._handle, table, keys, priorities, name=scope)

  def dataset(self,
              table: str,
              dtypes: Sequence[Any],
              shapes: Sequence[Any],
              capacity: int = 100,
              num_workers_per_iterator: int = -1,
              max_samples_per_stream: int = -1,
              sequence_length: Optional[int] = None,
              emit_timesteps: bool = True) -> ReplayDataset:
    """Creates a ReplayDataset which samples from Replay service.

    Note: Uses of Python lists are converted into tuples as nest used by the
    tf.data API doesn't have good support for lists.

    See ReplayDataset for detailed documentation.

    Args:
      table: Probability table to sample from.
      dtypes: Dtypes of the data output. Can be nested.
      shapes: Shapes of the data output. Can be nested. When `emit_timesteps`
        is True this is the shape of a single timestep in the sampled items;
        when it is False shapes must include `sequence_length`.
      capacity: (Defaults to 100) Maximum number of samples requested by the
        workers with each request. Higher values give higher throughput but too
        big values can result in skewed sampling distributions as large number
        of samples are fetched from single snapshot of the replay (followed by a
        period of lower activity as the samples are consumed). A good rule of
        thumb is to set this value to 2-3x times the batch size used.
      num_workers_per_iterator: (Defaults to -1, i.e auto selected) The number
        of worker threads to create per dataset iterator. When the selected
        table uses a FIFO sampler (i.e a queue) then exactly 1 worker must be
        used to avoid races causing invalid ordering of items. For all other
        samplers, this value should be roughly equal to the number of threads
        available on the CPU.
      max_samples_per_stream: (Defaults to -1, i.e auto selected) The maximum
        number of samples to fetch from a stream before a new call is made.
        Keeping this number low ensures that the data is fetched uniformly from
        all server.
      sequence_length: (Defaults to None, i.e unknown) The number of timesteps
        that each sample consists of. If set then the length of samples received
        from the server will be validated against this number.
      emit_timesteps: (Defaults to True) If set, timesteps instead of full
        sequences are retturned from the dataset. Returning sequences instead
        of timesteps can be more efficient as the memcopies caused by the
        splitting and batching of tensor can be avoided. Note that if set to
        False then then all `shapes` must have dim[0] equal to
        `sequence_length`.

    Returns:
      A ReplayDataset with the above specification.
    """
    return ReplayDataset(
        server_address=self._server_address,
        table=table,
        dtypes=dtypes,
        shapes=shapes,
        max_in_flight_samples_per_worker=capacity,
        num_workers_per_iterator=num_workers_per_iterator,
        max_samples_per_stream=max_samples_per_stream,
        sequence_length=sequence_length,
        emit_timesteps=emit_timesteps)


def _convert_lists_to_tuples(structure: Any) -> Any:
  list_to_tuple_fn = lambda s: tuple(s) if isinstance(s, list) else s
  # Traverse depth-first, bottom-up
  return tree.traverse(list_to_tuple_fn, structure, top_down=False)


def _is_tf1_runtime() -> bool:
  """Returns True if the runtime is executing with TF1.0 APIs."""
  # TODO(b/145023272): Update when/if there is a better way.
  return hasattr(tf, 'to_float')
