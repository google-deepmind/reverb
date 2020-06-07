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

"""Tests for tf_client."""

from concurrent import futures
import threading
import time

from absl.testing import parameterized
import numpy as np
from reverb import client as reverb_client
from reverb import item_selectors
from reverb import rate_limiters
from reverb import replay_sample
from reverb import server
from reverb import tf_client
import tensorflow.compat.v1 as tf
import tree

from tensorflow.python.framework import tensor_spec  # pylint:disable=g-direct-tensorflow-import


def make_server():
  return server.Server(
      tables=[
          server.Table(
              'dist',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1)),
          server.Table(
              'dist2',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1)),
          server.Table(
              'signatured',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1),
              signature=tf.TensorSpec(dtype=tf.float32, shape=(None, None))),
          server.Table(
              'bounded_spec_signatured',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1),
              # Currently only the `shape` and `dtype` of the bounded spec
              # is considered during signature check.
              # TODO(b/158033101): Check the boundaries as well.
              signature=tensor_spec.BoundedTensorSpec(
                  dtype=tf.float32,
                  shape=(None, None),
                  minimum=(0.0, 0.0),
                  maximum=(10.0, 10.)),
          ),
      ],
      port=None,
  )


class SampleOpTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def test_sets_meta_data_fields(self):
    input_data = [np.ones((81, 81), dtype=np.float64)]
    self._client.insert(input_data, {'dist': 1})
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      sample = session.run(client.sample('dist', [tf.float64]))
      np.testing.assert_equal(input_data, sample.data)
      self.assertNotEqual(sample.info.key, 0)
      self.assertEqual(sample.info.probability, 1)
      self.assertEqual(sample.info.table_size, 1)
      self.assertEqual(sample.info.priority, 1)

  def test_dtype_mismatch_result_in_error_raised(self):
    data = [np.zeros((81, 81))]
    self._client.insert(data, {'dist': 1})
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      with self.assertRaises(tf.errors.InternalError):
        session.run(client.sample('dist', [tf.float32]))

  def test_forwards_server_error(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      with self.assertRaises(tf.errors.NotFoundError):
        session.run(client.sample('invalid', [tf.float64]))

  def test_retries_until_success_or_fatal_error(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      with futures.ThreadPoolExecutor(max_workers=1) as executor:
        sample = executor.submit(session.run,
                                 client.sample('dist', [tf.float64]))
        input_data = [np.zeros((81, 81))]
        self._client.insert(input_data, {'dist': 1})
        np.testing.assert_equal(input_data, sample.result().data)


class UpdatePrioritiesOpTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def test_shape_result_in_error_raised(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      update_op = client.update_priorities(
          tf.constant('dist'), tf.constant([1, 2], dtype=tf.uint64),
          tf.constant([1], dtype=tf.float64))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        session.run(update_op)

  def test_priority_update_is_applied(self):
    # Start with uniform distribution
    for i in range(4):
      self._client.insert([np.array([i], dtype=np.uint32)], {'dist': 1})

    # Until we have recieved all 4 items.
    items = {}
    while len(items) < 4:
      item = next(self._client.sample('dist'))[0]
      items[item.info.key] = item.info.probability
      self.assertEqual(item.info.probability, 0.25)

    # Update the priority of one of the items.
    update_key = next(iter(items.keys()))
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      update_op = client.update_priorities(
          table=tf.constant('dist'),
          keys=tf.constant([update_key], dtype=tf.uint64),
          priorities=tf.constant([3], dtype=tf.float64))
      self.assertEqual(None, session.run(update_op))

    # The updated item now has priority 3 and the other 3 items have priority 1
    # each. The probability of sampling the new item should thus be 50%. We
    # sample until the updated item is seen and check that the probability (and
    # thus the priority) has been updated.
    for _ in range(1000):
      item = next(self._client.sample('dist'))[0]
      if item.info.key == update_key:
        self.assertEqual(item.info.probability, 0.5)
        break
    else:
      self.fail('Updated item was not found')


class InsertOpTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')
    self._client.reset('dist2')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def setUp(self):
    super().setUp()
    self.data = [tf.constant([1, 2, 3], dtype=tf.int8)]

  def test_checks_that_table_has_rank_1(self):
    client = tf_client.TFClient(self._client.server_address)
    priorities = tf.constant([1.0], dtype=tf.float64)

    # Works for rank 1.
    client.insert(self.data, tf.constant(['dist']), priorities)

    # Does not work for rank > 1.
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant([['dist']]), priorities)

    # Does not work for rank < 1.
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant('dist'), priorities)

  def test_checks_dtype_of_table_argument(self):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant([1]),
                    tf.constant([1.0], dtype=tf.float64))

  def test_checks_that_priorities_argument_has_rank_1(self):
    client = tf_client.TFClient(self._client.server_address)
    data = [tf.constant([1, 2])]
    tables = tf.constant(['dist'])

    # Works for rank 1.
    client.insert(data, tables, tf.constant([1.0], dtype=tf.float64))

    # Does not work for rank > 1.
    with self.assertRaises(ValueError):
      client.insert(data, tables, tf.constant([[1.0]], dtype=tf.float64))

    # Does not work for rank < 1.
    with self.assertRaises(ValueError):
      client.insert(data, tables, tf.constant(1.0, dtype=tf.float64))

  def test_checks_that_priorities_argument_has_dtype_float64(self):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant(['dist']),
                    tf.constant([1.0], dtype=tf.float32))

  def test_checks_that_tables_and_priorities_arguments_have_same_shape(self):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant(['dist', 'dist2']),
                    tf.constant([1.0], dtype=tf.float64))

  def test_single_table_insert(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      insert_op = client.insert(
          data=[tf.constant([1, 2, 3], dtype=tf.int8)],
          tables=tf.constant(['dist']),
          priorities=tf.constant([1.0], dtype=tf.float64))
      sample_op = client.sample('dist', [tf.int8])

      # Check that insert op succeeds.
      self.assertEqual(None, session.run(insert_op))

      # Check that the sampled data matches the inserted.
      sample = session.run(sample_op)
      self.assertLen(sample.data, 1)
      np.testing.assert_equal(
          np.array([1, 2, 3], dtype=np.int8), sample.data[0])

  def test_multi_table_insert(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      insert_op = client.insert(
          data=[tf.constant([1, 2, 3], dtype=tf.int8)],
          tables=tf.constant(['dist', 'dist2']),
          priorities=tf.constant([1.0, 2.0], dtype=tf.float64))

      sample_ops = [
          client.sample('dist', [tf.int8]),
          client.sample('dist2', [tf.int8])
      ]

      # Check that insert op succeeds.
      self.assertEqual(None, session.run(insert_op))

      # Check that the sampled data matches the inserted in all tables.
      for sample_op in sample_ops:
        sample = session.run(sample_op)
        self.assertLen(sample.data, 1)
        np.testing.assert_equal(
            np.array([1, 2, 3], dtype=np.int8), sample.data[0])


class DatasetTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')
    self._client.reset('signatured')
    self._client.reset('bounded_spec_signatured')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def _populate_replay(self, sequence_length=100, max_time_steps=None):
    max_time_steps = max_time_steps or sequence_length
    with self._client.writer(max_time_steps) as writer:
      for i in range(1000):
        writer.append([np.zeros((3, 3), dtype=np.float32)])
        if i % 5 == 0 and i >= sequence_length:
          writer.create_item(
              table='dist', num_timesteps=sequence_length, priority=1)
          writer.create_item(
              table='signatured', num_timesteps=sequence_length, priority=1)
          writer.create_item(
              table='bounded_spec_signatured',
              num_timesteps=sequence_length,
              priority=1)

  def _sample_from(self, dataset, num_samples):
    iterator = dataset.make_initializable_iterator()
    dataset_item = iterator.get_next()
    self.evaluate(iterator.initializer)
    return [self.evaluate(dataset_item) for _ in range(num_samples)]

  @parameterized.named_parameters(
      {
          'testcase_name': 'default_values',
      },
      {
          'testcase_name': 'num_workers_per_iterator_is_0',
          'num_workers_per_iterator': 0,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'num_workers_per_iterator_is_1',
          'num_workers_per_iterator': 1,
      },
      {
          'testcase_name': 'num_workers_per_iterator_is_minus_1',
          'num_workers_per_iterator': -1,
      },
      {
          'testcase_name': 'num_workers_per_iterator_is_minus_2',
          'num_workers_per_iterator': -2,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'max_samples_per_stream_is_0',
          'max_samples_per_stream': 0,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'max_samples_per_stream_is_1',
          'max_samples_per_stream': 1,
      },
      {
          'testcase_name': 'max_samples_per_stream_is_minus_1',
          'max_samples_per_stream': -1,
      },
      {
          'testcase_name': 'max_samples_per_stream_is_minus_2',
          'num_workers_per_iterator': -2,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'capacity_is_0',
          'capacity': 0,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'capacity_is_1',
          'capacity': 1,
      },
      {
          'testcase_name': 'capacity_is_minus_1',
          'capacity': -1,
          'want_error': ValueError,
      },
  )
  def test_sampler_parameter_validation(self, **kwargs):
    client = tf_client.TFClient(self._client.server_address)
    dtypes = (tf.float32,)
    shapes = (tf.TensorShape([3, 3]),)

    if 'want_error' in kwargs:
      error = kwargs.pop('want_error')
      with self.assertRaises(error):
        client.dataset('dist', dtypes, shapes, **kwargs)
    else:
      client.dataset('dist', dtypes, shapes, **kwargs)

  def test_iterate(self):
    self._populate_replay()

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist', dtypes=(tf.float32,), shapes=(tf.TensorShape([3, 3]),))
    got = self._sample_from(dataset, 10)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)
      # A single sample is returned so the key should be a scalar int64.
      self.assertIsInstance(sample.info.key, np.uint64)
      np.testing.assert_array_equal(sample.data[0],
                                    np.zeros((3, 3), dtype=np.float32))

  def test_timeout_invalid_arguments(self):
    client = tf_client.TFClient(self._client.server_address)

    with self.assertRaisesRegex(ValueError, r'must be an integer >= -1'):
      client.dataset(
          table='dist',
          dtypes=(tf.float32,),
          shapes=(tf.TensorShape([3, 3]),),
          rate_limiter_timeout_ms=-2)

  def test_timeout(self):
    client = tf_client.TFClient(self._client.server_address)

    dataset_0s = client.dataset(
        table='dist',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        rate_limiter_timeout_ms=0)

    dataset_1s = client.dataset(
        table='dist',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        rate_limiter_timeout_ms=1000)

    dataset_2s = client.dataset(
        table='dist',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        rate_limiter_timeout_ms=2000)

    start_time = time.time()
    with self.assertRaisesWithPredicateMatch(tf.errors.OutOfRangeError,
                                             r'End of sequence'):
      self._sample_from(dataset_0s, 1)
    duration = time.time() - start_time
    self.assertGreaterEqual(duration, 0)
    self.assertLess(duration, 5)

    start_time = time.time()
    with self.assertRaisesWithPredicateMatch(tf.errors.OutOfRangeError,
                                             r'End of sequence'):
      self._sample_from(dataset_1s, 1)
    duration = time.time() - start_time
    self.assertGreaterEqual(duration, 1)
    self.assertLess(duration, 10)

    start_time = time.time()
    with self.assertRaisesWithPredicateMatch(tf.errors.OutOfRangeError,
                                             r'End of sequence'):
      self._sample_from(dataset_2s, 1)
    duration = time.time() - start_time
    self.assertGreaterEqual(duration, 2)
    self.assertLess(duration, 10)

    # If we insert some data, and the rate limiter doesn't force any waiting,
    # then we can ask for a timeout of 0s and still get data back.
    self._populate_replay()
    got = self._sample_from(dataset_0s, 2)
    self.assertLen(got, 2)

  @parameterized.parameters(['signatured'], ['bounded_spec_signatured'])
  def test_inconsistent_signature_size(self, table_name):
    self._populate_replay()

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name,
        dtypes=(tf.float32, tf.float64),
        shapes=(tf.TensorShape([3, 3]), tf.TensorShape([])))
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Inconsistent number of tensors requested from table \'{}\'.  '
        r'Requested 6 tensors, but table signature shows 5 tensors.'.format(
            table_name)):
      self._sample_from(dataset, 10)

  @parameterized.parameters(['signatured'], ['bounded_spec_signatured'])
  def test_incomatible_signature_dtype(self, table_name):
    self._populate_replay()

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name, dtypes=(tf.int64,), shapes=(tf.TensorShape([3, 3]),))
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Requested incompatible tensor at flattened index 4 from table '
        r'\'{}\'.  Requested \(dtype, shape\): \(int64, \[3,3\]\).  '
        r'Signature \(dtype, shape\): \(float, \[\?,\?\]\)'.format(table_name)):
      self._sample_from(dataset, 10)

  @parameterized.parameters(['signatured'], ['bounded_spec_signatured'])
  def test_incompatible_signature_shape(self, table_name):
    self._populate_replay()

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name, dtypes=(tf.float32,), shapes=(tf.TensorShape([3]),))
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Requested incompatible tensor at flattened index 4 from table '
        r'\'{}\'.  Requested \(dtype, shape\): \(float, \[3\]\).  '
        r'Signature \(dtype, shape\): \(float, \[\?,\?\]\)'.format(table_name)):
      self._sample_from(dataset, 10)

  @parameterized.parameters([1], [3], [10])
  def test_incompatible_shape_when_using_sequence_length(self, sequence_length):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.dataset(
          table='dist',
          dtypes=(tf.float32,),
          shapes=(tf.TensorShape([sequence_length + 1, 3, 3]),),
          emit_timesteps=False,
          sequence_length=sequence_length)

  @parameterized.parameters(
      ('dist', 1, 1),
      ('dist', 1, 3),
      ('dist', 3, 3),
      ('dist', 3, 5),
      ('dist', 10, 10),
      ('dist', 10, 11),
      ('signatured', 1, 1),
      ('signatured', 3, 3),
      ('signatured', 3, 5),
      ('signatured', 10, 10),
      ('bounded_spec_signatured', 1, 1),
      ('bounded_spec_signatured', 3, 3),
      ('bounded_spec_signatured', 3, 5),
      ('bounded_spec_signatured', 10, 10),
  )
  def test_iterate_with_sequence_length(self, table_name, sequence_length,
                                        max_time_steps):
    # Also ensure we get sequence_length-shaped outputs when
    # writers' max_time_steps != sequence_length.
    self._populate_replay(sequence_length, max_time_steps=max_time_steps)

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([sequence_length, 3, 3]),),
        emit_timesteps=False,
        sequence_length=sequence_length)

    got = self._sample_from(dataset, 10)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)

      # The keys and data should be batched up by the sequence length.
      self.assertEqual(sample.info.key.shape, (sequence_length,))
      np.testing.assert_array_equal(
          sample.data[0], np.zeros((sequence_length, 3, 3), dtype=np.float32))

  @parameterized.parameters(
      ('dist', 1),
      ('dist', 3),
      ('dist', 10),
      ('signatured', 1),
      ('signatured', 3),
      ('signatured', 10),
      ('bounded_spec_signatured', 1),
      ('bounded_spec_signatured', 3),
      ('bounded_spec_signatured', 10),
  )
  def test_iterate_with_unknown_sequence_length(self, table_name,
                                                sequence_length):
    self._populate_replay(sequence_length)

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([None, 3, 3]),),
        emit_timesteps=False,
        sequence_length=None)

    # Check the shape of the items.
    iterator = dataset.make_initializable_iterator()
    dataset_item = iterator.get_next()
    self.assertIsNone(dataset_item.info.key.shape.as_list()[0], None)
    self.assertIsNone(dataset_item.data[0].shape.as_list()[0], None)

    # Verify that once evaluated, the samples has the expected length.
    got = self._sample_from(dataset, 10)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)

      # The keys and data should be batched up by the sequence length.
      self.assertEqual(sample.info.key.shape, (sequence_length,))
      np.testing.assert_array_equal(
          sample.data[0], np.zeros((sequence_length, 3, 3), dtype=np.float32))

  @parameterized.parameters(
      ('dist', 1, 2),
      ('dist', 2, 1),
      ('signatured', 1, 2),
      ('signatured', 2, 1),
      ('bounded_spec_signatured', 1, 2),
      ('bounded_spec_signatured', 2, 1),
  )
  def test_checks_sequence_length_when_timesteps_emitted(
      self, table_name, actual_sequence_length, provided_sequence_length):
    self._populate_replay(actual_sequence_length)

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([provided_sequence_length, 3, 3]),),
        emit_timesteps=True,
        sequence_length=provided_sequence_length)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self._sample_from(dataset, 10)

  @parameterized.named_parameters(
      dict(testcase_name='TableDist', table_name='dist'),
      dict(testcase_name='TableSignatured', table_name='signatured'),
      dict(
          testcase_name='TableBoundedSpecSignatured',
          table_name='bounded_spec_signatured'))
  def test_iterate_batched(self, table_name):
    self._populate_replay()

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),))
    dataset = dataset.batch(2, True)

    got = self._sample_from(dataset, 10)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)

      # The keys should be batched up like the data.
      self.assertEqual(sample.info.key.shape, (2,))

      np.testing.assert_array_equal(sample.data[0],
                                    np.zeros((2, 3, 3), dtype=np.float32))

  def test_iterate_nested_and_batched(self):
    with self._client.writer(100) as writer:
      for i in range(1000):
        writer.append({
            'observation': {
                'data': np.zeros((3, 3), dtype=np.float32),
                'extras': [
                    np.int64(10),
                    np.ones([1], dtype=np.int32),
                ],
            },
            'reward': np.zeros((10, 10), dtype=np.float32),
        })
        if i % 5 == 0 and i >= 100:
          writer.create_item(
              table='dist', num_timesteps=100, priority=1)

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist',
        dtypes=(((tf.float32), (tf.int64, tf.int32)), tf.float32),
        shapes=((tf.TensorShape([3, 3]), (tf.TensorShape(None),
                                          tf.TensorShape([1]))),
                tf.TensorShape([10, 10])),
    )
    dataset = dataset.batch(3)

    structure = {
        'observation': {
            'data':
                tf.TensorSpec([3, 3], tf.float32),
            'extras': [
                tf.TensorSpec([], tf.int64),
                tf.TensorSpec([1], tf.int32),
            ],
        },
        'reward': tf.TensorSpec([], tf.int64),
    }

    got = self._sample_from(dataset, 10)
    self.assertLen(got, 10)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)

      transition = tree.unflatten_as(structure, tree.flatten(sample.data))
      np.testing.assert_array_equal(transition['observation']['data'],
                                    np.zeros([3, 3, 3], dtype=np.float32))
      np.testing.assert_array_equal(transition['observation']['extras'][0],
                                    np.ones([3], dtype=np.int64) * 10)
      np.testing.assert_array_equal(transition['observation']['extras'][1],
                                    np.ones([3, 1], dtype=np.int32))
      np.testing.assert_array_equal(transition['reward'],
                                    np.zeros([3, 10, 10], dtype=np.float32))

  def test_multiple_iterators(self):
    with self._client.writer(100) as writer:
      for i in range(10):
        writer.append([np.ones((81, 81), dtype=np.float32) * i])
      writer.create_item(table='dist', num_timesteps=10, priority=1)

    trajectory_length = 5
    batch_size = 3

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist', dtypes=(tf.float32,), shapes=(tf.TensorShape([81, 81]),))
    dataset = dataset.batch(trajectory_length)

    iterators = [
        dataset.make_initializable_iterator() for _ in range(batch_size)
    ]
    items = tf.stack(
        [tf.squeeze(iterator.get_next().data) for iterator in iterators])

    with self.session() as session:
      session.run([iterator.initializer for iterator in iterators])
      got = session.run(items)
      self.assertEqual(got.shape, (batch_size, trajectory_length, 81, 81))

      want = np.array(
          [[np.ones([81, 81]) * i for i in range(trajectory_length)]] *
          batch_size)
      np.testing.assert_array_equal(got, want)

  def test_iterate_over_blobs(self):
    for _ in range(10):
      self._client.insert((np.ones([3, 3], dtype=np.int32)), {'dist': 1})

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist', dtypes=(tf.int32,), shapes=(tf.TensorShape([3, 3]),))

    got = self._sample_from(dataset, 20)
    self.assertLen(got, 20)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)
      self.assertIsInstance(sample.info.key, np.uint64)
      self.assertIsInstance(sample.info.probability, np.float64)
      np.testing.assert_array_equal(sample.data[0],
                                    np.ones((3, 3), dtype=np.int32))

  def test_iterate_over_batched_blobs(self):
    for _ in range(10):
      self._client.insert((np.ones([3, 3], dtype=np.int32)), {'dist': 1})

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist', dtypes=(tf.int32,), shapes=(tf.TensorShape([3, 3]),))

    dataset = dataset.batch(5)

    got = self._sample_from(dataset, 20)
    self.assertLen(got, 20)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)
      self.assertEqual(sample.info.key.shape, (5,))
      np.testing.assert_array_equal(sample.data[0],
                                    np.ones((5, 3, 3), dtype=np.int32))

  def test_converts_spec_lists_into_tuples(self):
    for _ in range(10):
      data = [
          (np.ones([1, 1], dtype=np.int32),),
          [
              np.ones([3, 3], dtype=np.int8),
              (np.ones([2, 2], dtype=np.float64),)
          ],
      ]
      self._client.insert(data, {'dist': 1})

    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist',
        dtypes=[
            (tf.int32,),
            [
                tf.int8,
                (tf.float64,),
            ],
        ],
        shapes=[
            (tf.TensorShape([1, 1]),),
            [
                tf.TensorShape([3, 3]),
                (tf.TensorShape([2, 2]),),
            ],
        ])

    got = self._sample_from(dataset, 10)

    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)
      self.assertIsInstance(sample.info.key, np.uint64)
      tree.assert_same_structure(sample.data, (
          (None,),
          (
              None,
              (None,),
          ),
      ))

  def test_session_is_closed_while_op_pending(self):
    client = tf_client.TFClient(self._client.server_address)
    dataset = client.dataset(
        table='dist', dtypes=tf.float32, shapes=tf.TensorShape([]))

    iterator = dataset.make_initializable_iterator()
    item = iterator.get_next()

    def _session_closer(sess, wait_time_secs):
      def _fn():
        time.sleep(wait_time_secs)
        sess.close()

      return _fn

    with self.session() as sess:
      sess.run(iterator.initializer)
      thread = threading.Thread(target=_session_closer(sess, 3))
      thread.start()
      with self.assertRaises(tf.errors.CancelledError):
        sess.run(item)


if __name__ == '__main__':
  tf.disable_eager_execution()
  tf.test.main()
