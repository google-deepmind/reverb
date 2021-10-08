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

"""Tests for dataset."""

import socket
import threading
import time

from absl.testing import parameterized
import numpy as np
from reverb import client
from reverb import dataset as reverb_dataset
from reverb import errors
from reverb import item_selectors
from reverb import rate_limiters
from reverb import replay_sample
from reverb import server as reverb_server
import tensorflow.compat.v1 as tf
import tree

from tensorflow.python.framework import tensor_spec  # pylint:disable=g-direct-tensorflow-import


def make_server():
  return reverb_server.Server(
      tables=[
          reverb_server.Table(
              'dist',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1)),
          reverb_server.Table(
              'dist_queue',
              sampler=item_selectors.Fifo(),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              max_times_sampled=1,
              rate_limiter=rate_limiters.MinSize(1)),
          reverb_server.Table(
              'signatured',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1),
              signature=tf.TensorSpec(dtype=tf.float32, shape=(None, None))),
          reverb_server.Table(
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


class LocalReplayDatasetTest(tf.test.TestCase, parameterized.TestCase):
  USE_LOCALHOST = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    if cls.USE_LOCALHOST:
      connect_to = 'localhost'
    else:
      connect_to = 'dns:///{}'.format(socket.gethostname())
    cls._client = client.Client(f'{connect_to}:{cls._server.port}')

  def setUp(self):
    super().setUp()
    self._num_prev_samples = {
        table: self._get_total_num_samples(table)
        for table in ('dist', 'dist_queue', 'signatured',
                      'bounded_spec_signatured')
    }

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')
    self._client.reset('dist_queue')
    self._client.reset('signatured')
    self._client.reset('bounded_spec_signatured')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def _populate_replay(self,
                       sequence_length=100,
                       max_time_steps=None,
                       max_items=1000):
    max_time_steps = max_time_steps or sequence_length
    num_items = 0
    with self._client.writer(max_time_steps) as writer:
      for i in range(1000):
        writer.append([np.zeros((3, 3), dtype=np.float32)])
        if i % min(5, sequence_length) == 0 and i >= sequence_length:
          writer.create_item(
              table='dist', num_timesteps=sequence_length, priority=1)
          writer.create_item(
              table='dist_queue', num_timesteps=sequence_length, priority=1)
          writer.create_item(
              table='signatured', num_timesteps=sequence_length, priority=1)
          writer.create_item(
              table='bounded_spec_signatured',
              num_timesteps=sequence_length,
              priority=1)
          num_items += 1
          if num_items >= max_items:
            break

  def _sample_from(self, dataset, num_samples):
    iterator = dataset.make_initializable_iterator()
    dataset_item = iterator.get_next()
    self.evaluate(iterator.initializer)
    return [self.evaluate(dataset_item) for _ in range(num_samples)]

  def _get_total_num_samples(self, table: str) -> int:
    table_info = self._client.server_info()[table]
    return table_info.rate_limiter_info.sample_stats.completed

  def _get_num_samples(self, table: str) -> int:
    """Gets the number of samples since the start of the test."""
    return self._get_total_num_samples(table) - self._num_prev_samples[table]

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
          'testcase_name': 'max_in_flight_samples_per_worker_is_0',
          'max_in_flight_samples_per_worker': 0,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'max_in_flight_samples_per_worker_is_1',
          'max_in_flight_samples_per_worker': 1,
      },
      {
          'testcase_name': 'max_in_flight_samples_per_worker_is_minus_1',
          'max_in_flight_samples_per_worker': -1,
          'want_error': ValueError,
      },
  )
  def test_sampler_parameter_validation(self, **kwargs):
    dtypes = (tf.float32,)
    shapes = (tf.TensorShape([3, 3]),)

    if 'max_in_flight_samples_per_worker' not in kwargs:
      kwargs['max_in_flight_samples_per_worker'] = 100

    if 'want_error' in kwargs:
      error = kwargs.pop('want_error')
      with self.assertRaises(error):
        reverb_dataset.ReplayDataset(self._client.server_address, 'dist',
                                     dtypes, shapes, **kwargs)
    else:
      reverb_dataset.ReplayDataset(self._client.server_address, 'dist', dtypes,
                                   shapes, **kwargs)

  def test_iterate(self):
    self._populate_replay()

    dataset = reverb_dataset.ReplayDataset(
        tf.constant(self._client.server_address),
        table=tf.constant('dist'),
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        max_in_flight_samples_per_worker=100)
    got = self._sample_from(dataset, 10)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)
      # A single sample is returned so the key should be a scalar int64.
      self.assertIsInstance(sample.info.key, np.uint64)
      np.testing.assert_array_equal(sample.data[0],
                                    np.zeros((3, 3), dtype=np.float32))

  def test_distribution_strategy(self):
    self._populate_replay()

    physical_devices = tf.config.list_physical_devices('CPU')

    configs = tf.config.experimental.get_virtual_device_configuration(
        physical_devices[0])
    if configs is None:
      virtual_devices = [tf.config.experimental.VirtualDeviceConfiguration()
                         for _ in range(4)]
      tf.config.experimental.set_virtual_device_configuration(
          physical_devices[0], virtual_devices)

    strategy = tf.distribute.MirroredStrategy(['/cpu:%d' % i for i in range(4)])

    def reverb_dataset_fn(i):
      tf.print('Creating dataset for replica; index:', i)
      return reverb_dataset.ReplayDataset(
          self._client.server_address,
          table=tf.constant('dist'),
          dtypes=(tf.float32,),
          shapes=(tf.TensorShape([3, 3]),),
          max_in_flight_samples_per_worker=100).take(2)

    def dataset_fn(_):
      return tf.data.Dataset.range(4).flat_map(reverb_dataset_fn).take(2 * 4)

    ds = strategy.experimental_distribute_datasets_from_function(dataset_fn)

    def check_probabilities(_, v):
      probability = v.info.probability
      self.assertLen(probability.values, 4)
      # Don't use any math ops since tensor values seem to contain
      # unaligned tensors on some systems; but tf.print doesn't check alignment.
      #
      # This seems to be caused by a compatibility issue where DistStrat isn't
      # well tested when eager mode is disabled.  So instead of treating this
      # as a true TF bug, we just work around it.  We can remove this hack and
      # convert it to e.g. tf.assert_greater type check if/when we enable eager
      # execution for these tests.
      tf.print('Probability values:', probability.values)

    def get_next_value(v):
      return tf.distribute.get_replica_context().merge_call(
          check_probabilities, args=(v,))

    @tf.function
    def run_strategy(ds_):
      i = tf.constant(0)
      for v in ds_:
        strategy.run(get_next_value, args=(v,))
        i += 1
      return i

    rs = run_strategy(ds)

    # Each iteration contains 4 items - one from each replica.  We take 8 items
    # total, so there should be 2 iterations.
    self.assertEqual(2, self.evaluate(rs))

  def test_timeout_invalid_arguments(self):
    with self.assertRaisesRegex(ValueError, r'must be an integer >= -1'):
      reverb_dataset.ReplayDataset(
          self._client.server_address,
          table='dist',
          dtypes=(tf.float32,),
          shapes=(tf.TensorShape([3, 3]),),
          rate_limiter_timeout_ms=-2,
          max_in_flight_samples_per_worker=100)

  def test_timeout(self):
    dataset_0s = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist_queue',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        rate_limiter_timeout_ms=50,  # Slightly above exactly 0.
        max_in_flight_samples_per_worker=100)

    dataset_1s = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist_queue',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        rate_limiter_timeout_ms=1000,
        max_in_flight_samples_per_worker=100)

    dataset_2s = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist_queue',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        rate_limiter_timeout_ms=2000,
        max_in_flight_samples_per_worker=100)

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
    iterator = dataset_0s.make_initializable_iterator()
    dataset_0s_item = iterator.get_next()
    self.evaluate(iterator.initializer)

    for _ in range(3):
      self._populate_replay(max_items=2)
      # Pull two items
      for _ in range(2):
        self.evaluate(dataset_0s_item)
      # Wait for the time it would take a broken sampler to time out
      # on next iteration.
      time.sleep(0.5)

  @parameterized.parameters(['signatured'], ['bounded_spec_signatured'])
  def test_inconsistent_signature_size(self, table_name):
    self._populate_replay()

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32, tf.float64),
        shapes=(tf.TensorShape([3, 3]), tf.TensorShape([])),
        max_in_flight_samples_per_worker=100)
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Inconsistent number of tensors requested from table \'{}\'.  '
        r'Requested 6 tensors, but table signature shows 5 tensors.'.format(
            table_name)):
      self._sample_from(dataset, 10)

  @parameterized.parameters(['signatured'], ['bounded_spec_signatured'])
  def test_incompatible_signature_dtype(self, table_name):
    self._populate_replay()
    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.int64,),
        shapes=(tf.TensorShape([3, 3]),),
        max_in_flight_samples_per_worker=100)
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Requested incompatible tensor at flattened index 4 from table '
        r'\'{}\'.  Requested \(dtype, shape\): \(int64, \[3,3\]\).  '
        r'Signature \(dtype, shape\): \(float, \[\?,\?\]\)'.format(table_name)):
      self._sample_from(dataset, 10)

    dataset_emit_sequences = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.int64,),
        shapes=(tf.TensorShape([None, 3, 3]),),
        emit_timesteps=False,
        max_in_flight_samples_per_worker=100)
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Requested incompatible tensor at flattened index 4 from table '
        r'\'{}\'.  Requested \(dtype, shape\): \(int64, \[3,3\]\).  '
        r'Signature \(dtype, shape\): \(float, \[\?,\?\]\)'.format(table_name)):
      self._sample_from(dataset_emit_sequences, 10)

  @parameterized.parameters(['signatured'], ['bounded_spec_signatured'])
  def test_incompatible_signature_shape(self, table_name):
    self._populate_replay()

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3]),),
        max_in_flight_samples_per_worker=100)
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Requested incompatible tensor at flattened index 4 from table '
        r'\'{}\'.  Requested \(dtype, shape\): \(float, \[3\]\).  '
        r'Signature \(dtype, shape\): \(float, \[\?,\?\]\)'.format(table_name)):
      self._sample_from(dataset, 10)

    dataset_emit_sequences = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([None, 3]),),
        emit_timesteps=False,
        max_in_flight_samples_per_worker=100)
    with self.assertRaisesWithPredicateMatch(
        tf.errors.InvalidArgumentError,
        r'Requested incompatible tensor at flattened index 4 from table '
        r'\'{}\'.  Requested \(dtype, shape\): \(float, \[3\]\).  '
        r'Signature \(dtype, shape\): \(float, \[\?,\?\]\)'.format(table_name)):
      self._sample_from(dataset_emit_sequences, 10)

  @parameterized.parameters([1], [3], [10])
  def test_incompatible_shape_when_using_sequence_length(self, sequence_length):
    with self.assertRaises(ValueError):
      reverb_dataset.ReplayDataset(
          self._client.server_address,
          table='dist',
          dtypes=(tf.float32,),
          shapes=(tf.TensorShape([sequence_length + 1, 3, 3]),),
          emit_timesteps=False,
          sequence_length=sequence_length,
          max_in_flight_samples_per_worker=100)

  def test_incompatible_dataset_shapes_and_types_without_signature(self):
    self._populate_replay()
    ds_wrong_shape = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([]),),
        max_in_flight_samples_per_worker=100)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'Specification has \(dtype, shape\): \(float, \[\]\).  '
        r'Tensor has \(dtype, shape\): \(float, \[3,3\]\).'):
      self._sample_from(ds_wrong_shape, 1)

    ds_full_sequences_wrong_shape = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([None]),),
        emit_timesteps=False,
        max_in_flight_samples_per_worker=100)

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r'Specification has \(dtype, shape\): \(float, \[\]\).  '
        r'Tensor has \(dtype, shape\): \(float, \[3,3\]\).'):
      self._sample_from(ds_full_sequences_wrong_shape, 1)

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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([sequence_length, 3, 3]),),
        emit_timesteps=False,
        sequence_length=sequence_length,
        max_in_flight_samples_per_worker=100)

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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([None, 3, 3]),),
        emit_timesteps=False,
        sequence_length=None,
        max_in_flight_samples_per_worker=100)

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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([provided_sequence_length, 3, 3]),),
        emit_timesteps=True,
        sequence_length=provided_sequence_length,
        max_in_flight_samples_per_worker=100)

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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table=table_name,
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([3, 3]),),
        max_in_flight_samples_per_worker=100)
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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(((tf.float32), (tf.int64, tf.int32)), tf.float32),
        shapes=((tf.TensorShape([3, 3]), (tf.TensorShape(None),
                                          tf.TensorShape([1]))),
                tf.TensorShape([10, 10])),
        max_in_flight_samples_per_worker=100)
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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(tf.float32,),
        shapes=(tf.TensorShape([81, 81]),),
        max_in_flight_samples_per_worker=100)
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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(tf.int32,),
        shapes=(tf.TensorShape([3, 3]),),
        max_in_flight_samples_per_worker=100)

    got = self._sample_from(dataset, 20)
    self.assertLen(got, 20)
    for sample in got:
      self.assertIsInstance(sample, replay_sample.ReplaySample)
      self.assertIsInstance(sample.info.key, np.uint64)
      self.assertIsInstance(sample.info.probability, np.float64)
      np.testing.assert_array_equal(sample.data[0],
                                    np.ones((3, 3), dtype=np.int32))

  @parameterized.parameters(1, 3, 7)
  def test_respects_max_in_flight_samples_per_worker(
      self, max_in_flight_samples_per_worker):
    if not self.USE_LOCALHOST:
      self.skipTest('TODO(b/190761815): test broken in Nonlocal case')

    for _ in range(10):
      self._client.insert((np.ones([3, 3], dtype=np.int32)), {'dist': 1})

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(tf.int32,),
        shapes=(tf.TensorShape([3, 3]),),
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker)

    iterator = dataset.make_initializable_iterator()
    dataset_item = iterator.get_next()
    self.evaluate(iterator.initializer)

    for _ in range(100):
      self.evaluate(dataset_item)

      # Check that the buffer is incremented by steps of
      # max_in_flight_samples_per_worker.
      self.assertEqual(
          self._get_num_samples('dist') % max_in_flight_samples_per_worker, 0)

  def test_iterate_over_batched_blobs(self):
    for _ in range(10):
      self._client.insert((np.ones([3, 3], dtype=np.int32)), {'dist': 1})

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=(tf.int32,),
        shapes=(tf.TensorShape([3, 3]),),
        max_in_flight_samples_per_worker=100)

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

    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
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
        ],
        max_in_flight_samples_per_worker=100)

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
    dataset = reverb_dataset.ReplayDataset(
        self._client.server_address,
        table='dist',
        dtypes=tf.float32,
        shapes=tf.TensorShape([]),
        max_in_flight_samples_per_worker=100)

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


class NonlocalReplayDatasetTest(LocalReplayDatasetTest):
  """Test with non-localhost connection to server."""
  USE_LOCALHOST = False


class FromTableSignatureTest(tf.test.TestCase):

  def test_table_not_found(self):
    server = reverb_server.Server([
        reverb_server.Table.queue('table_a', 10),
        reverb_server.Table.queue('table_c', 10),
        reverb_server.Table.queue('table_b', 10),
    ])
    address = f'localhost:{server.port}'

    with self.assertRaisesWithPredicateMatch(
        ValueError,
        f'Server at {address} does not contain any table named not_found. '
        f'Found: table_a, table_b, table_c.'):
      reverb_dataset.ReplayDataset.from_table_signature(
          address, 'not_found', 100)

  def test_server_not_found(self):
    with self.assertRaises(errors.DeadlineExceededError):
      reverb_dataset.ReplayDataset.from_table_signature(
          'localhost:1234', 'not_found', 100, get_signature_timeout_secs=1)

  def test_table_does_not_have_signature(self):
    server = make_server()
    address = f'localhost:{server.port}'
    with self.assertRaisesWithPredicateMatch(
        ValueError, f'Table dist at {address} does not have a signature.'):
      reverb_dataset.ReplayDataset.from_table_signature(
          address, 'dist', 100)

  def test_sets_dtypes_from_signature(self):
    signature = {
        'a': {
            'b': tf.TensorSpec([3, 3], tf.float32),
            'c': tf.TensorSpec([], tf.int64),
        },
        'x': tf.TensorSpec([None], tf.uint64),
    }

    server = reverb_server.Server(
        [reverb_server.Table.queue('queue', 10, signature=signature)])

    dataset = reverb_dataset.ReplayDataset.from_table_signature(
        f'localhost:{server.port}', 'queue', 100)
    self.assertDictEqual(dataset.element_spec.data, signature)

  def test_sets_dtypes_from_bounded_spec_signature(self):
    bounded_spec_signature = {
        'a': {
            'b': tensor_spec.BoundedTensorSpec([3, 3], tf.float32, 0, 3),
            'c': tensor_spec.BoundedTensorSpec([], tf.int64, 0, 5),
        },
    }

    server = reverb_server.Server([
        reverb_server.Table.queue(
            'queue', 10, signature=bounded_spec_signature)
    ])

    dataset = reverb_dataset.ReplayDataset.from_table_signature(
        f'localhost:{server.port}', 'queue', 100)
    self.assertDictEqual(
        dataset.element_spec.data, {
            'a': {
                'b': tf.TensorSpec([3, 3], tf.float32),
                'c': tf.TensorSpec([], tf.int64),
            },
        })

  def test_combines_sequence_length_with_signature_if_not_emit_timestamps(self):
    server = reverb_server.Server([
        reverb_server.Table.queue(
            'queue',
            10,
            signature={
                'a': {
                    'b': tf.TensorSpec([3, 3], tf.float32),
                    'c': tf.TensorSpec([], tf.int64),
                },
            })
    ])

    dataset = reverb_dataset.ReplayDataset.from_table_signature(
        f'localhost:{server.port}',
        'queue',
        100,
        emit_timesteps=False,
        sequence_length=5)
    self.assertDictEqual(
        dataset.element_spec.data, {
            'a': {
                'b': tf.TensorSpec([5, 3, 3], tf.float32),
                'c': tf.TensorSpec([5], tf.int64),
            },
        })


if __name__ == '__main__':
  tf.disable_eager_execution()
  tf.test.main()
