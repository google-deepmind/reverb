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

"""Tests for python client."""

import collections
import multiprocessing.dummy as multithreading
import pickle

from absl.testing import absltest
import numpy as np
from reverb import client
from reverb import errors
from reverb import item_selectors
from reverb import rate_limiters
from reverb import server
import tensorflow.compat.v1 as tf

TABLE_NAME = 'table'
NO_SIGNATURE_TABLE = 'no_signature'


class ClientTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.server = server.Server(
        tables=[
            server.Table(
                name=TABLE_NAME,
                sampler=item_selectors.Prioritized(1),
                remover=item_selectors.Fifo(),
                max_size=1000,
                rate_limiter=rate_limiters.MinSize(3),
                signature=tf.TensorSpec(dtype=tf.int64, shape=()),
            ),
            server.Table.queue(NO_SIGNATURE_TABLE, 10),
        ],
        port=None)
    cls.client = client.Client(f'localhost:{cls.server.port}')

  def tearDown(self):
    self.client.reset(TABLE_NAME)
    super().tearDown()

  @classmethod
  def tearDownClass(cls):
    cls.server.stop()
    super().tearDownClass()

  def _get_sample_frequency(self, n=10000):
    keys = [sample[0].info.key for sample in self.client.sample(TABLE_NAME, n)]
    counter = collections.Counter(keys)
    return [count / n for _, count in counter.most_common()]

  def test_sample_sets_table_size(self):
    for i in range(1, 11):
      self.client.insert(i, {TABLE_NAME: 1.0})
      if i >= 3:
        sample = next(self.client.sample(TABLE_NAME, 1))[0]
        self.assertEqual(sample.info.table_size, i)

  def test_sample_sets_probability(self):
    for i in range(1, 11):
      self.client.insert(i, {TABLE_NAME: 1.0})
      if i >= 3:
        sample = next(self.client.sample(TABLE_NAME, 1))[0]
        self.assertAlmostEqual(sample.info.probability, 1.0 / i, 0.01)

  def test_sample_sets_priority(self):
    # Set the test context by manually mutating priorities to known ones.
    for i in range(10):
      self.client.insert(i, {TABLE_NAME: 1000.0})

    def _sample_priorities(n=100):
      return {
          sample[0].info.key: sample[0].info.priority
          for sample in self.client.sample(TABLE_NAME, n)
      }

    original_priorities = _sample_priorities(n=100)
    self.assertNotEmpty(original_priorities)
    self.assertSequenceAlmostEqual([1000.0] * len(original_priorities),
                                   original_priorities.values())
    expected_priorities = {
        key: float(i) for i, key in enumerate(original_priorities)
    }
    self.client.mutate_priorities(TABLE_NAME, updates=expected_priorities)

    # Resample and check priorities.
    sampled_priorities = _sample_priorities(n=100)
    self.assertNotEmpty(sampled_priorities)
    for key, priority in sampled_priorities.items():
      if key in expected_priorities:
        self.assertAlmostEqual(expected_priorities[key], priority)

  def test_insert_raises_if_priorities_empty(self):
    with self.assertRaises(ValueError):
      self.client.insert([1], {})

  def test_insert(self):
    self.client.insert(1, {TABLE_NAME: 1.0})  # This should be sampled often.
    self.client.insert(2, {TABLE_NAME: 0.1})  # This should be sampled rarely.
    self.client.insert(3, {TABLE_NAME: 0.0})  # This should never be sampled.

    freqs = self._get_sample_frequency()

    self.assertLen(freqs, 2)
    self.assertAlmostEqual(freqs[0], 0.9, delta=0.05)
    self.assertAlmostEqual(freqs[1], 0.1, delta=0.05)

  def test_writer_raises_if_max_sequence_length_lt_1(self):
    with self.assertRaises(ValueError):
      self.client.writer(0)

  def test_writer_raises_if_chunk_length_lt_1(self):
    self.client.writer(2, chunk_length=1)  # Should be fine.

    for chunk_length in [0, -1]:
      with self.assertRaises(ValueError):
        self.client.writer(2, chunk_length=chunk_length)

  def test_writer_raises_if_chunk_length_gt_max_sequence_length(self):
    self.client.writer(2, chunk_length=1)  # lt should be fine.
    self.client.writer(2, chunk_length=2)  # eq should be fine.

    with self.assertRaises(ValueError):
      self.client.writer(2, chunk_length=3)

  def test_writer_raises_if_max_in_flight_items_lt_1(self):
    self.client.writer(1, max_in_flight_items=1)
    self.client.writer(1, max_in_flight_items=2)
    self.client.writer(1, max_in_flight_items=None)

    with self.assertRaises(ValueError):
      self.client.writer(1, max_in_flight_items=-1)

  def test_writer_works_with_no_retries(self):
    # If the server responds correctly, the writer ignores the no retries arg.
    writer = self.client.writer(2)
    writer.append([0])
    writer.create_item(TABLE_NAME, 1, 1.0)
    writer.close(retry_on_unavailable=False)

  def test_writer(self):
    with self.client.writer(2) as writer:
      writer.append([0])
      writer.create_item(TABLE_NAME, 1, 1.0)
      writer.append([1])
      writer.create_item(TABLE_NAME, 2, 1.0)
      writer.append([2])
      writer.create_item(TABLE_NAME, 1, 1.0)
      writer.append_sequence([np.array([3, 4])])
      writer.create_item(TABLE_NAME, 2, 1.0)

    freqs = self._get_sample_frequency()
    self.assertLen(freqs, 4)
    for freq in freqs:
      self.assertAlmostEqual(freq, 0.25, delta=0.05)

  def test_write_and_sample_different_shapes_and_dtypes(self):
    trajectories = [
        np.ones([], np.int64),
        np.ones([2, 2], np.float32),
        np.ones([3, 3], np.int32),
    ]
    for trajectory in trajectories:
      self.client.insert(trajectory, {NO_SIGNATURE_TABLE: 1.0})

    for i, [sample] in enumerate(self.client.sample(NO_SIGNATURE_TABLE, 3)):
      np.testing.assert_array_equal(trajectories[i], sample.data[0])

  def test_mutate_priorities_update(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    before = self._get_sample_frequency()
    self.assertLen(before, 3)
    for freq in before:
      self.assertAlmostEqual(freq, 0.33, delta=0.05)

    key = next(self.client.sample(TABLE_NAME, 1))[0].info.key
    self.client.mutate_priorities(TABLE_NAME, updates={key: 0.5})

    after = self._get_sample_frequency()
    self.assertLen(after, 3)
    self.assertAlmostEqual(after[0], 0.4, delta=0.05)
    self.assertAlmostEqual(after[1], 0.4, delta=0.05)
    self.assertAlmostEqual(after[2], 0.2, delta=0.05)

  def test_mutate_priorities_delete(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    before = self._get_sample_frequency()
    self.assertLen(before, 4)

    key = next(self.client.sample(TABLE_NAME, 1))[0].info.key
    self.client.mutate_priorities(TABLE_NAME, deletes=[key])

    after = self._get_sample_frequency()
    self.assertLen(after, 3)

  def test_reset(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    keys_before = set(
        sample[0].info.key for sample in self.client.sample(TABLE_NAME, 1000))
    self.assertLen(keys_before, 3)

    self.client.reset(TABLE_NAME)

    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})

    keys_after = set(
        sample[0].info.key for sample in self.client.sample(TABLE_NAME, 1000))
    self.assertLen(keys_after, 3)

    self.assertTrue(keys_after.isdisjoint(keys_before))

  def test_server_info(self):
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    self.client.insert([0], {TABLE_NAME: 1.0})
    server_info = self.client.server_info()
    self.assertLen(server_info, 2)

    self.assertIn(TABLE_NAME, server_info)
    info = server_info[TABLE_NAME]
    self.assertEqual(info.current_size, 3)
    self.assertEqual(info.max_size, 1000)
    self.assertEqual(info.sampler_options.prioritized.priority_exponent, 1)
    self.assertTrue(info.remover_options.fifo)
    self.assertEqual(info.signature, tf.TensorSpec(dtype=tf.int64, shape=()))

    self.assertIn(NO_SIGNATURE_TABLE, server_info)
    info = server_info[NO_SIGNATURE_TABLE]
    self.assertEqual(info.current_size, 0)
    self.assertEqual(info.max_size, 10)
    self.assertTrue(info.sampler_options.fifo)
    self.assertTrue(info.remover_options.fifo)
    self.assertIsNone(info.signature)

  def test_server_info_timeout(self):
    # Setup a client that doesn't actually connect to anything.
    dummy_client = client.Client(f'localhost:{self.server.port + 1}')
    with self.assertRaises(
        errors.DeadlineExceededError,
        msg='ServerInfo call did not complete within provided timeout of 1s'):
      dummy_client.server_info(timeout=1)

  def test_pickle(self):
    loaded_client = pickle.loads(pickle.dumps(self.client))
    self.assertEqual(loaded_client._server_address, self.client._server_address)
    loaded_client.insert([0], {TABLE_NAME: 1.0})

  def test_multithreaded_writer_using_flush(self):
    # Ensure that we don't have any errors caused by multithreaded use of
    # writers or clients.
    pool = multithreading.Pool(64)
    def _write(i):
      with self.client.writer(1) as writer:
        writer.append([i])
        # Make sure that flush before create_item doesn't create trouble.
        writer.flush()
        writer.create_item(TABLE_NAME, 1, 1.0)
        writer.flush()

    for _ in range(5):
      pool.map(_write, list(range(256)))

    info = self.client.server_info()[TABLE_NAME]
    self.assertEqual(info.current_size, 1000)
    pool.close()
    pool.join()

  def test_multithreaded_writer_using_scope(self):
    # Ensure that we don't have any errors caused by multithreaded use of
    # writers or clients.
    pool = multithreading.Pool(64)
    def _write(i):
      with self.client.writer(1) as writer:
        writer.append([i])
        writer.create_item(TABLE_NAME, 1, 1.0)

    for _ in range(5):
      pool.map(_write, list(range(256)))

    info = self.client.server_info()[TABLE_NAME]
    self.assertEqual(info.current_size, 1000)
    pool.close()
    pool.join()

  def test_validates_trajectory_writer_config(self):
    with self.assertRaises(ValueError):
      self.client.trajectory_writer(0)

    with self.assertRaises(ValueError):
      self.client.trajectory_writer(-1)


if __name__ == '__main__':
  absltest.main()
