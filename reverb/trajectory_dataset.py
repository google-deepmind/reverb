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

"""Dataset to sample items created using the TrajectoryWriter."""

from typing import Any, List, Optional, Union

from reverb import client as reverb_client
from reverb import replay_sample
import tensorflow.compat.v1 as tf
import tree

from reverb.cc.ops import gen_reverb_ops


class TrajectoryDataset(tf.data.Dataset):
  """A tf.data.Dataset which samples trajectories from a Reverb table.

  Note: The dataset returns `ReplaySample` where `data` with the structure of
  `dtypes` and `shapes` and where all fields within `info` are scalars.

  Note: Uses of Python lists are converted into tuples as nest used by the
  tf.data API doesn't have good support for lists.
  """

  def __init__(self,
               server_address: Union[str, tf.Tensor],
               table: Union[str, tf.Tensor],
               dtypes: Any,
               shapes: Any,
               max_in_flight_samples_per_worker: int,
               num_workers_per_iterator: int = -1,
               max_samples_per_stream: int = -1,
               rate_limiter_timeout_ms: int = -1,
               max_samples: int = -1):
    """Constructs a new TrajectoryDataset.

    Args:
      server_address: Address of gRPC ReverbService.
      table: Probability table to sample from.
      dtypes: Dtypes of the data output. Can be nested.
      shapes: Shapes of the data output. Can be nested.
      max_in_flight_samples_per_worker: The number of samples requested in each
        batch of samples. Higher values give higher throughput but too big
        values can result in skewed sampling distributions as large number of
        samples are fetched from single snapshot of the replay (followed by a
        period of lower activity as the samples are consumed). A good rule of
        thumb is to set this value to 2-3x times the batch size used.
      num_workers_per_iterator: (Defaults to -1, i.e. auto selected) The number
        of worker threads to create per dataset iterator. When the selected
        table uses a FIFO sampler (i.e. a queue) then exactly 1 worker must be
        used to avoid races causing invalid ordering of items. For all other
        samplers, this value should be roughly equal to the number of threads
        available on the CPU.
      max_samples_per_stream: (Defaults to -1, i.e. auto selected) The maximum
        number of samples to fetch from a stream before a new call is made.
        Keeping this number low ensures that the data is fetched uniformly from
        all server.
      rate_limiter_timeout_ms: (Defaults to -1: infinite). Timeout (in
        milliseconds) to wait on the rate limiter when sampling from the table.
        If `rate_limiter_timeout_ms >= 0`, this is the timeout passed to
        `Table::Sample` describing how long to wait for the rate limiter to
          allow sampling. The first time that a request times out (across any of
          the workers), the Dataset iterator is closed and the sequence is
          considered finished.
      max_samples: (Defaults to -1: infinite). The maximum number of samples to
        request from the server. Once target number of samples has been fetched
        and returned, the iterator is closed. This can be used to avoid the
        prefetched added by the dataset.

    Raises:
      ValueError: If `dtypes` and `shapes` don't share the same structure.
      ValueError: If `max_in_flight_samples_per_worker` is not a
        positive integer.
      ValueError: If `num_workers_per_iterator` is not a positive integer or -1.
      ValueError: If `max_samples_per_stream` is not a positive integer or -1.
      ValueError: If `rate_limiter_timeout_ms < -1`.
      ValueError: If `max_samples` is not a positive integer or -1.

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
    if rate_limiter_timeout_ms < -1:
      raise ValueError('rate_limiter_timeout_ms (%d) must be an integer >= -1' %
                       rate_limiter_timeout_ms)
    if max_samples < 1 and max_samples != -1:
      raise ValueError('max_samples (%d) must be a positive integer or -1' %
                       max_samples)

    # Add the info fields (all scalars).
    dtypes = replay_sample.ReplaySample(
        info=replay_sample.SampleInfo.tf_dtypes(), data=dtypes)
    shapes = replay_sample.ReplaySample(
        info=replay_sample.SampleInfo.tf_shapes(), data=shapes)

    # The tf.data API doesn't fully support lists so we convert all uses of
    # lists into tuples.
    dtypes = _convert_lists_to_tuples(dtypes)
    shapes = _convert_lists_to_tuples(shapes)

    self._server_address = server_address
    self._table = table
    self._dtypes = dtypes
    self._shapes = shapes
    self._max_in_flight_samples_per_worker = max_in_flight_samples_per_worker
    self._num_workers_per_iterator = num_workers_per_iterator
    self._max_samples_per_stream = max_samples_per_stream
    self._rate_limiter_timeout_ms = rate_limiter_timeout_ms
    self._max_samples = max_samples

    if _is_tf1_runtime():
      # Disabling to avoid errors given the different tf.data.Dataset init args
      # between v1 and v2 APIs.
      # pytype: disable=wrong-arg-count
      super().__init__()
    else:
      # DatasetV2 requires the dataset as a variant tensor during init.
      super().__init__(self._as_variant_tensor())
      # pytype: enable=wrong-arg-count

  @classmethod
  def from_table_signature(cls,
                           server_address: str,
                           table: str,
                           max_in_flight_samples_per_worker: int,
                           num_workers_per_iterator: int = -1,
                           max_samples_per_stream: int = -1,
                           rate_limiter_timeout_ms: int = -1,
                           get_signature_timeout_secs: Optional[int] = None,
                           max_samples: int = -1):
    """Constructs a TrajectoryDataset using the table's signature to infer specs.

    Note: The target `Table` must specify a signature which represent the entire
      trajectory (as opposed to a single timestep). See `Table.__init__`
      (./server.py) for more details.

    Args:
      server_address: Address of gRPC ReverbService.
      table: Table to read the signature and sample from.
      max_in_flight_samples_per_worker: See __init__ for details.
      num_workers_per_iterator: See __init__ for details.
      max_samples_per_stream: See __init__ for details.
      rate_limiter_timeout_ms: See __init__ for details.
      get_signature_timeout_secs: Timeout in seconds to wait for server to
        respond when fetching the table signature. By default no timeout is set
        and the call will block indefinitely if the server does not respond.
      max_samples: See __init__ for details.

    Returns:
      TrajectoryDataset using the specs defined by the table signature to build
        `shapes` and `dtypes`.

    Raises:
      ValueError: If `table` does not exist on server at `server_address`.
      ValueError: If `table` does not have a signature.
      errors.DeadlineExceededError: If `get_signature_timeout_secs` provided and
        exceeded.
      ValueError: See __init__.
    """
    client = reverb_client.Client(server_address)
    info = client.server_info(get_signature_timeout_secs)
    if table not in info:
      raise ValueError(
          f'Server at {server_address} does not contain any table named '
          f'{table}. Found: {", ".join(sorted(info.keys()))}.')

    if not info[table].signature:
      raise ValueError(
          f'Table {table} at {server_address} does not have a signature.')

    shapes = tree.map_structure(lambda x: x.shape, info[table].signature)
    dtypes = tree.map_structure(lambda x: x.dtype, info[table].signature)

    return cls(
        server_address=server_address,
        table=table,
        shapes=shapes,
        dtypes=dtypes,
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
        num_workers_per_iterator=num_workers_per_iterator,
        max_samples_per_stream=max_samples_per_stream,
        rate_limiter_timeout_ms=rate_limiter_timeout_ms,
        max_samples=max_samples)

  def _as_variant_tensor(self):
    return gen_reverb_ops.reverb_trajectory_dataset(
        server_address=self._server_address,
        table=self._table,
        dtypes=tree.flatten(self._dtypes),
        shapes=tree.flatten(self._shapes),
        max_in_flight_samples_per_worker=self._max_in_flight_samples_per_worker,
        num_workers_per_iterator=self._num_workers_per_iterator,
        max_samples_per_stream=self._max_samples_per_stream,
        rate_limiter_timeout_ms=self._rate_limiter_timeout_ms,
        max_samples=self._max_samples)

  def _inputs(self) -> List[Any]:
    return []

  @property
  def element_spec(self) -> Any:
    return tree.map_structure(tf.TensorSpec, self._shapes, self._dtypes)


def _convert_lists_to_tuples(structure: Any) -> Any:
  list_to_tuple_fn = lambda s: tuple(s) if isinstance(s, list) else s
  # Traverse depth-first, bottom-up
  return tree.traverse(list_to_tuple_fn, structure, top_down=False)


def _is_tf1_runtime() -> bool:
  """Returns True if the runtime is executing with TF1.0 APIs."""
  # TODO(b/145023272): Update when/if there is a better way.
  return hasattr(tf, 'to_float')
