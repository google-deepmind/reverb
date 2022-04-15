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

from typing import Optional, Sequence

from reverb import replay_sample
import tensorflow.compat.v1 as tf
import tree

from reverb.cc.ops import gen_reverb_ops


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
    self._handle = gen_reverb_ops.reverb_client(
        server_address=server_address, shared_name=shared_name, name=name)

  @property
  def server_address(self) -> str:
    return self._server_address

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
      key, probability, table_size, priority, times_sampled, data = (
          gen_reverb_ops.reverb_client_sample(
              self._handle, table, tree.flatten(data_dtypes), name=scope))
      return replay_sample.ReplaySample(
          info=replay_sample.SampleInfo(
              key=key,
              probability=probability,
              table_size=table_size,
              priority=priority,
              times_sampled=times_sampled),
          data=tree.unflatten_as(data_dtypes, data))

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
      return gen_reverb_ops.reverb_client_insert(
          self._handle, data, tables, priorities, name=scope)

  def update_priorities(self,
                        table: str,
                        keys: tf.Tensor,
                        priorities: tf.Tensor,
                        name: Optional[str] = None):
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
      return gen_reverb_ops.reverb_client_update_priorities(
          self._handle, table, keys, priorities, name=scope)
