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

"""Python bindings for creating and serving the Reverb ReverbService.

See ./client.py and ./tf_client.py for details of how to interact with the
service.
"""

import abc
import collections
from typing import List, Optional, Sequence, Union

from absl import logging
import portpicker
from reverb import checkpointer as checkpointer_lib
from reverb import client
from reverb import item_selectors
from reverb import pybind
from reverb import rate_limiters
from reverb import reverb_types
import termcolor
import tree

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import tensor_spec
from tensorflow.python.saved_model import nested_structure_coder
# pylint: enable=g-direct-tensorflow-import


class TableExtensionBase(metaclass=abc.ABCMeta):
  """Abstract base class for Table extensions."""

  @abc.abstractmethod
  def build_internal_extensions(
      self, table_name: str) -> List[pybind.TableExtensionInterface]:
    """Constructs the c++ PriorityTableExtensions."""


# TODO(b/156334283): Delete this alias.
PriorityTableExtensionBase = TableExtensionBase


class Table:
  # TODO(b/157149247): Improve docstring.
  """Table defines how items are selected for sampling and removal."""

  def __init__(self,
               name: str,
               sampler: reverb_types.DistributionType,
               remover: reverb_types.DistributionType,
               max_size: int,
               rate_limiter: rate_limiters.RateLimiter,
               max_times_sampled: int = 0,
               extensions: Sequence[TableExtensionBase] = (),
               signature: Optional[reverb_types.SpecNest] = None):
    """Constructor of the Table.

    Args:
      name: Name of the priority table.
      sampler: The strategy to use when selecting samples.
      remover: The strategy to use when selecting which items to remove.
      max_size: The maximum number of items which the replay is allowed to hold.
        When an item is inserted into an already full priority table the
        `remover` is used for selecting which item to remove before proceeding
        with the new insert.
      rate_limiter: Manages the data flow by limiting the sample and insert
        calls.
      max_times_sampled: Maximum number of times an item can be sampled before
        it is deleted. Any value < 1 is ignored and means there is no limit.
      extensions: Optional sequence of extensions used to add extra features to
        the table.
      signature: Optional nested structure containing `tf.TypeSpec` objects,
        describing the storage schema for this table.

    Raises:
      ValueError: If name is empty.
      ValueError: If max_size <= 0.
    """
    if not name:
      raise ValueError('name must be nonempty')
    if max_size <= 0:
      raise ValueError('max_size (%d) must be a positive integer' % max_size)

    # Merge the c++ extensions into a single list.
    internal_extensions = []
    for extension in extensions:
      internal_extensions += extension.build_internal_extensions(name)

    if signature:
      flat_signature = tree.flatten(signature)
      for s in flat_signature:
        if not isinstance(s, tensor_spec.TensorSpec):
          raise ValueError(f'Unsupported signature spec: {s}')
      signature_proto_str = (
          nested_structure_coder.StructureCoder().encode_structure(
              signature).SerializeToString())
    else:
      signature_proto_str = None

    self.internal_table = pybind.Table(
        name=name,
        sampler=sampler,
        remover=remover,
        max_size=max_size,
        max_times_sampled=max_times_sampled,
        rate_limiter=rate_limiter.internal_limiter,
        extensions=internal_extensions,
        signature=signature_proto_str)

  @classmethod
  def queue(cls, name: str, max_size: int):
    """Constructs a Table which acts like a queue.

    Args:
      name: Name of the priority table (aka queue).
      max_size: Maximum number of items in the priority table (aka queue).

    Returns:
      Table which behaves like a queue of size `max_size`.
    """
    return cls(
        name=name,
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=1,
        rate_limiter=rate_limiters.Queue(max_size))

  @classmethod
  def stack(cls, name: str, max_size: int):
    """Constructs a Table which acts like a stack.

    Args:
      name: Name of the priority table (aka stack).
      max_size: Maximum number of items in the priority table (aka stack).

    Returns:
      Table which behaves like a stack of size `max_size`.
    """
    return cls(
        name=name,
        sampler=item_selectors.Lifo(),
        remover=item_selectors.Lifo(),
        max_size=max_size,
        max_times_sampled=1,
        rate_limiter=rate_limiters.Stack(max_size))

  @property
  def name(self):
    return self.internal_table.name()

  def can_sample(self, num_samples: int) -> bool:
    """Returns True if a sample operation is permitted at the current state."""
    return self.internal_table.can_sample(num_samples)

  def can_insert(self, num_inserts: int) -> bool:
    """Returns True if an insert operation is permitted at the current state."""
    return self.internal_table.can_insert(num_inserts)


# TODO(b/156334283): Delete this alias.
PriorityTable = Table


class Server:
  """Reverb replay server.

  The Server hosts the gRPC-service deepmind.reverb.ReverbService (see
  //third_party/reverb/reverb_service.proto). See ./client.py and ./tf_client
  for details of how to interact with the service.

  A Server maintains inserted data and one or more PriorityTables. Multiple
  tables can be used to provide different views of the same underlying and since
  the operations performed by the Table is relatively inexpensive compared to
  operations on the actual data using multiple tables referencing the same data
  is encouraged over replicating data.
  """

  def __init__(self,
               priority_tables: List[Table],
               port: Union[int, None],
               checkpointer: checkpointer_lib.CheckpointerBase = None):
    """Constructor of Server serving the ReverbService.

    Args:
      priority_tables: A list of priority tables to host on the server.
      port: The port number to serve the gRPC-service on. If `None` is passed
        then a port is automatically picked and assigned.
      checkpointer: Checkpointer used for storing/loading checkpoints. If None
        (default) then `checkpointer_lib.default_checkpointer` is used to
        construct the checkpointer.

    Raises:
      ValueError: If priority_tables is empty.
      ValueError: If multiple Table in priority_tables share names.
    """
    if not priority_tables:
      raise ValueError('At least one priority table must be provided')
    names = collections.Counter(table.name for table in priority_tables)
    duplicates = [name for name, count in names.items() if count > 1]
    if duplicates:
      raise ValueError(
          'Multiple items in priority_tables have the same name: {}'.format(
              ', '.join(duplicates)))

    if port is None:
      port = portpicker.pick_unused_port()

    if checkpointer is None:
      checkpointer = checkpointer_lib.default_checkpointer()

    self._server = pybind.Server(
        [table.internal_table for table in priority_tables], port,
        checkpointer.internal_checkpointer())
    self._port = port

  def __del__(self):
    """Stop server and free up the port if was reserved through portpicker."""
    if hasattr(self, '_server'):
      self.stop()

    if hasattr(self, '_port'):
      portpicker.return_port(self._port)

  @property
  def port(self):
    """Port the gRPC service is running at."""
    return self._port

  def stop(self):
    """Request that the ReverbService is terminated and wait for shutdown."""
    return self._server.Stop()

  def wait(self):
    """Wait indefinitely for the ReverbService to stop."""
    return self._server.Wait()

  def in_process_client(self):
    """Gets a local in process client.

    This bypasses proto serialization and network overhead.

    Returns:
      Client. Must not be used after this ReplayServer has been stopped!
    """
    return client.Client(f'[::1]:{self._port}', self._server.InProcessClient())
