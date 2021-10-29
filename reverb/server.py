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
from typing import Optional, Sequence, Union

from absl import logging
import portpicker
from reverb import client
from reverb import item_selectors
from reverb import pybind
from reverb import rate_limiters
from reverb import reverb_types
from reverb.platform.default import checkpointers

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
      self,
      table_name: str,
  ) -> Sequence[pybind.TableExtension]:
    """Constructs the c++ PriorityTableExtensions."""


class Table:
  """Item collection with configurable strategies for insertion and sampling.

  A `Table` is the structure used to interact with the data stored on a server.
  Each table can contain a limited number of "items" that can be retrieved
  according to the strategy defined by the `sampler`. The size of a table, in
  terms of number of items, is limited to `max_size`. When items are inserted
  into an already full table the `remover` is used to decide which item should
  be removed.

  In addition to the selection strategies used to select items for retrieval and
  removal the flow of data is controlled by a `RateLimiter`. A rate limiter
  controlls high level relations between inserts and samples by defining a
  target ratio between the two and what level of deviations from the target is
  acceptable. This is particularily useful when scaling up from single machine
  use cases to distributed systems as the same "logical" throughput can be kept
  constant even though the scale has changed by orders of magnitude.

  It is important to note that "data elements" and "items" are related but
  distinct types of entities.

    Data element:
      - The actual data written using `Writer.append`.
      - Immutable once written.
      - Is not stored in a `Table`.
      - Can be referenced by items from one or more distinct `Table`.
      - Cannot be retrieved in any other way than as a part of an item.

    Item:
      - The entity stored in a `Table`.
      - Inserted using `Writer.create_item`.
      - References one or more data elements, creating a "sequence".

  The fact that data elements can be referenced by more than one item from one
  or multiple tables means thats one has to be careful not to equate the size of
  a table (in terms of items) with the amount of data it references. The data
  will remain in memory on the server until the last item that references it is
  removed from its table. Removing an item from a table does therefore not
  neccesarily result in any (significant) change in memory usage and one must be
  careful when selecting remover strategies for a multi table server. Consider
  for example a server with two tables. One has a FIFO remover and the other
  LIFO remover. In this scenario, the two tables would not share any chunks and
  would eventually consume twice the amount of memory compared a similar setup
  where the two tables share the same type of removal strategy.
  """

  def __init__(self,
               name: str,
               sampler: reverb_types.SelectorType,
               remover: reverb_types.SelectorType,
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
        describing the schema of items in this table.

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
      internal_extensions += list(extension.build_internal_extensions(name))

    if signature:
      flat_signature = tree.flatten(signature)
      for s in flat_signature:
        if not isinstance(s, tensor_spec.TensorSpec):
          raise ValueError(f'Unsupported signature spec: {s}')
      signature_proto_str = (
          nested_structure_coder.encode_structure(
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
  def queue(cls,
            name: str,
            max_size: int,
            extensions: Sequence[TableExtensionBase] = (),
            signature: Optional[reverb_types.SpecNest] = None):
    """Constructs a Table which acts like a queue.

    Args:
      name: Name of the priority table (aka queue).
      max_size: Maximum number of items in the priority table (aka queue).
      extensions: See documentation in the constructor.
      signature: See documentation in the constructor.

    Returns:
      Table which behaves like a queue of size `max_size`.
    """
    return cls(
        name=name,
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=1,
        rate_limiter=rate_limiters.Queue(max_size),
        extensions=extensions,
        signature=signature)

  @classmethod
  def stack(cls,
            name: str,
            max_size: int,
            extensions: Sequence[TableExtensionBase] = (),
            signature: Optional[reverb_types.SpecNest] = None):
    """Constructs a Table which acts like a stack.

    Args:
      name: Name of the priority table (aka stack).
      max_size: Maximum number of items in the priority table (aka stack).
      extensions: See documentation in the constructor.
      signature: See documentation in the constructor.

    Returns:
      Table which behaves like a stack of size `max_size`.
    """
    return cls(
        name=name,
        sampler=item_selectors.Lifo(),
        remover=item_selectors.Lifo(),
        max_size=max_size,
        max_times_sampled=1,
        rate_limiter=rate_limiters.Stack(max_size),
        extensions=extensions,
        signature=signature)

  @property
  def name(self):
    return self.internal_table.name()

  @property
  def info(self) -> reverb_types.TableInfo:
    proto_string = self.internal_table.info()
    return reverb_types.TableInfo.from_serialized_proto(proto_string)

  def can_sample(self, num_samples: int) -> bool:
    """Returns True if a sample operation is permitted at the current state."""
    return self.internal_table.can_sample(num_samples)

  def can_insert(self, num_inserts: int) -> bool:
    """Returns True if an insert operation is permitted at the current state."""
    return self.internal_table.can_insert(num_inserts)

  def __repr__(self):
    return repr(self.internal_table)


class Server:
  """Reverb replay server.

  The Server hosts the gRPC-service deepmind.reverb.ReverbService (see
  reverb_service.proto). See ./client.py and ./tf_client for details of how to
  interact with the service.

  A Server maintains inserted data and one or more PriorityTables. Multiple
  tables can be used to provide different views of the same underlying and since
  the operations performed by the Table is relatively inexpensive compared to
  operations on the actual data using multiple tables referencing the same data
  is encouraged over replicating data.
  """

  def __init__(self,
               tables: Optional[Sequence[Table]] = None,
               port: Optional[Union[int, None]] = None,
               checkpointer: Optional[checkpointers.CheckpointerBase] = None):
    """Constructor of Server serving the ReverbService.

    Args:
      tables: A sequence of tables to host on the server.
      port: The port number to serve the gRPC-service on. If `None` (default)
        then a port is automatically picked and assigned.
      checkpointer: Checkpointer used for storing/loading checkpoints. If None
        (default) then `checkpointers.default_checkpointer` is used to
        construct the checkpointer.

    Raises:
      ValueError: If tables is empty.
      ValueError: If multiple Table in tables share names.
    """
    if not tables:
      raise ValueError('At least one table must be provided')
    names = collections.Counter(table.name for table in tables)
    duplicates = [name for name, count in names.items() if count > 1]
    if duplicates:
      raise ValueError('Multiple items in tables have the same name: {}'.format(
          ', '.join(duplicates)))

    if port is None:
      port = portpicker.pick_unused_port()

    if checkpointer is None:
      checkpointer = checkpointers.default_checkpointer()

    self._server = pybind.Server([table.internal_table for table in tables],
                                 port, checkpointer.internal_checkpointer())
    self._port = port

  def __del__(self):
    """Stop server and free up the port if was reserved through portpicker."""
    if hasattr(self, '_server'):
      self.stop()

    if hasattr(self, '_port'):
      portpicker.return_port(self._port)

  def __repr__(self):
    return repr(self._server)

  @property
  def port(self):
    """Port the gRPC service is running at."""
    return self._port

  def stop(self):
    """Request that the ReverbService is terminated and wait for shutdown."""
    return self._server.Stop()

  def wait(self):
    """Blocks until the service is shut down.

    This method will never return unless the server is shut down which will only
    happen if:

      * `Server.stop` is called by another thread.
      * A KeyboardInterrupt is raised (i.e. a SIGINT signal is sent to the
        process).

    Raises:
      KeyboardInterrupt: If the server was killed by a SIGINT.
    """
    if self._server.Wait():
      raise KeyboardInterrupt

  def localhost_client(self):
    """Creates a client connect to the localhost channel."""
    return client.Client(f'localhost:{self._port}')
