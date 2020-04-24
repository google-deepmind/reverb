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

"""Tests for python server.

Note: Most of the functionality is tested through ./client_test.py. This file
only contains a few extra cases which does not fit well in the client tests.
"""

from absl.testing import absltest
from reverb import distributions
from reverb import rate_limiters
from reverb import server

TABLE_NAME = 'table'


class ServerTest(absltest.TestCase):

  def test_in_process_client(self):
    my_server = server.Server(
        priority_tables=[
            server.PriorityTable(
                name=TABLE_NAME,
                sampler=distributions.Prioritized(1),
                remover=distributions.Fifo(),
                max_size=100,
                rate_limiter=rate_limiters.MinSize(2)),
        ],
        port=None)
    my_client = my_server.in_process_client()
    my_client.reset(TABLE_NAME)
    del my_client
    my_server.stop()

  def test_duplicate_priority_table_name(self):
    with self.assertRaises(ValueError):
      server.Server(
          priority_tables=[
              server.PriorityTable(
                  name='test',
                  sampler=distributions.Prioritized(1),
                  remover=distributions.Fifo(),
                  max_size=100,
                  rate_limiter=rate_limiters.MinSize(2)),
              server.PriorityTable(
                  name='test',
                  sampler=distributions.Prioritized(2),
                  remover=distributions.Fifo(),
                  max_size=200,
                  rate_limiter=rate_limiters.MinSize(1))
          ],
          port=None)

  def test_no_priority_table_provided(self):
    with self.assertRaises(ValueError):
      server.Server(priority_tables=[], port=None)

  def test_can_sample(self):
    table = server.PriorityTable(
        name=TABLE_NAME,
        sampler=distributions.Prioritized(1),
        remover=distributions.Fifo(),
        max_size=100,
        max_times_sampled=1,
        rate_limiter=rate_limiters.MinSize(2))
    my_server = server.Server(priority_tables=[table], port=None)
    my_client = my_server.in_process_client()
    self.assertFalse(table.can_sample(1))
    self.assertTrue(table.can_insert(1))
    my_client.insert(1, {TABLE_NAME: 1.0})
    self.assertFalse(table.can_sample(1))
    my_client.insert(1, {TABLE_NAME: 1.0})
    self.assertTrue(table.can_sample(2))
    # TODO(b/153258711): This should return False since max_times_sampled=1.
    self.assertTrue(table.can_sample(3))
    del my_client
    my_server.stop()


if __name__ == '__main__':
  absltest.main()
