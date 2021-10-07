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

"""Tests for server_from_proto."""

from absl.testing import absltest
from absl.testing import parameterized
import reverb
from reverb.server_executable import server_from_proto

from reverb.cc import schema_pb2
from reverb.cc.checkpointing import checkpoint_pb2


class ServerFromProtoTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('fifo', 'fifo', reverb.selectors.Fifo),
      ('lifo', 'lifo', reverb.selectors.Lifo),
      ('uniform', 'uniform', reverb.selectors.Uniform))
  def test_selector_from_proto(self, selector_proto_field,
                               expected_selector):
    selector_proto = schema_pb2.KeyDistributionOptions()
    setattr(selector_proto, selector_proto_field, True)
    selector = server_from_proto.selector_from_proto(selector_proto)
    self.assertIsInstance(selector, expected_selector)

  def test_prioritized_selector_from_proto(self):
    selector_proto = schema_pb2.KeyDistributionOptions()
    selector_proto.prioritized.priority_exponent = 2
    selector = server_from_proto.selector_from_proto(selector_proto)
    self.assertIsInstance(selector, reverb.selectors.Prioritized)

  @parameterized.named_parameters(
      ('min_heap', True, reverb.selectors.MinHeap),
      ('max_heap', False, reverb.selectors.MaxHeap))
  def test_heap_selector_from_proto(self, is_min_heap,
                                    expected_selector_builder):
    selector_proto = schema_pb2.KeyDistributionOptions()
    selector_proto.heap.min_heap = is_min_heap
    selector = server_from_proto.selector_from_proto(selector_proto)
    self.assertIsInstance(selector, type(expected_selector_builder()))

  def test_rate_limiter_from_proto(self):
    rate_limiter_proto = checkpoint_pb2.RateLimiterCheckpoint()
    rate_limiter_proto.samples_per_insert = 1
    rate_limiter_proto.min_size_to_sample = 2
    rate_limiter_proto.min_diff = -100
    rate_limiter_proto.max_diff = 120
    rate_limiter = server_from_proto.rate_limiter_from_proto(
        rate_limiter_proto)
    self.assertIsInstance(rate_limiter,
                          reverb.rate_limiters.RateLimiter)

  def test_table_from_proto(self):
    table_proto = checkpoint_pb2.PriorityTableCheckpoint()
    table_proto.table_name = 'test_table'
    table_proto.max_size = 101
    table_proto.max_times_sampled = 200
    table_proto.rate_limiter.min_diff = -100
    table_proto.rate_limiter.max_diff = 200
    table_proto.rate_limiter.samples_per_insert = 10
    table_proto.rate_limiter.min_size_to_sample = 1
    table_proto.sampler.lifo = True
    table_proto.remover.fifo = True
    config = [
        table_proto
    ]
    tables = server_from_proto.tables_from_proto(config)
    self.assertLen(tables, 1)
    self.assertIsInstance(tables[0], reverb.Table)
    self.assertEqual(tables[0].name, table_proto.table_name)
    table_info = tables[0].info
    self.assertEqual(table_info.max_size, table_proto.max_size)
    self.assertEqual(table_info.max_times_sampled,
                     table_proto.max_times_sampled)
    # The rate limiter objects do not have quite the same structure.
    self.assertEqual(table_info.rate_limiter_info.max_diff,
                     table_proto.rate_limiter.max_diff)
    self.assertEqual(table_info.rate_limiter_info.min_diff,
                     table_proto.rate_limiter.min_diff)
    self.assertEqual(table_info.rate_limiter_info.samples_per_insert,
                     table_proto.rate_limiter.samples_per_insert)
    self.assertEqual(table_info.rate_limiter_info.min_size_to_sample,
                     table_proto.rate_limiter.min_size_to_sample)
    self.assertEqual(table_info.sampler_options.lifo,
                     table_proto.sampler.lifo)
    self.assertEqual(table_info.remover_options.fifo,
                     table_proto.remover.fifo)


if __name__ == '__main__':
  absltest.main()
