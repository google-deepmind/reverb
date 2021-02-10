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

"""Tests for trajectory_dataset."""

import importlib

from absl.testing import parameterized
import numpy as np
from reverb import client
from reverb import errors
from reverb import item_selectors
from reverb import rate_limiters
from reverb import replay_sample
from reverb import server as reverb_server
import tensorflow.compat.v1 as tf
import tree

from tensorflow.python.framework import tensor_spec  # pylint:disable=g-direct-tensorflow-import

# TODO(b/179907041): Replace with "from reverb import ...".
trajectory_writer = importlib.import_module('reverb.trajectory_writer')
trajectory_dataset = importlib.import_module('reverb.trajectory_dataset')

TABLE = 'prioritized'
DTYPES = {
    'observation': tf.float32,
    'reward': tf.int64,
}
SHAPES = {
    'observation': tf.TensorShape([1, 3, 3]),
    'reward': tf.TensorShape([]),
}


def make_server():
  return reverb_server.Server(
      tables=[
          reverb_server.Table(
              name=TABLE,
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000,
              rate_limiter=rate_limiters.MinSize(1)),
      ],
      port=None,
  )


class TrajectoryDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset(TABLE)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def _populate_replay(self):
    with trajectory_writer.TrajectoryWriter(self._client, 1, 1) as writer:
      for _ in range(10):
        writer.append([np.ones([3, 3], np.float32), 3])
        writer.create_item(TABLE, 1.0, {
            'observation': writer.history[0][-1:],
            'reward': writer.history[1][-1],
        })

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
      {
          'testcase_name': 'flexible_batch_size_is_minus_2',
          'flexible_batch_size': -2,
          'want_error': ValueError,
      },
      {
          'testcase_name': 'flexible_batch_size_is_0',
          'flexible_batch_size': 0,
          'want_error': ValueError,
      },
  )
  def test_sampler_parameter_validation(self, **kwargs):
    if 'max_in_flight_samples_per_worker' not in kwargs:
      kwargs['max_in_flight_samples_per_worker'] = 1

    if 'want_error' in kwargs:
      error = kwargs.pop('want_error')
      with self.assertRaises(error):
        trajectory_dataset.TrajectoryDataset(
            server_address=self._client.server_address,
            table=TABLE,
            dtypes=DTYPES,
            shapes=SHAPES,
            **kwargs)
    else:
      trajectory_dataset.TrajectoryDataset(
          server_address=self._client.server_address,
          table=TABLE,
          dtypes=DTYPES,
          shapes=SHAPES,
          **kwargs)

  def test_sample_fixed_length_trajectory(self):
    self._populate_replay()

    dataset = trajectory_dataset.TrajectoryDataset(
        tf.constant(self._client.server_address),
        table=tf.constant(TABLE),
        dtypes=DTYPES,
        shapes=SHAPES,
        max_in_flight_samples_per_worker=1,
        flexible_batch_size=1)

    tree.assert_same_structure(
        self._sample_from(dataset, 1)[0],
        replay_sample.ReplaySample(
            info=replay_sample.SampleInfo(
                key=1,
                probability=1.0,
                table_size=10,
                priority=0.5,
            ),
            data=SHAPES))

  def test_sample_variable_length_trajectory(self):
    with trajectory_writer.TrajectoryWriter(self._client, 2, 10) as writer:
      for i in range(10):
        writer.append([np.ones([3, 3], np.int32) * i])
        writer.create_item(TABLE, 1.0, {
            'last': writer.history[0][-1],
            'all': writer.history[0][:],
        })

    dataset = trajectory_dataset.TrajectoryDataset(
        tf.constant(self._client.server_address),
        table=tf.constant(TABLE),
        dtypes={
            'last': tf.int32,
            'all': tf.int32,
        },
        shapes={
            'last': tf.TensorShape([3, 3]),
            'all': tf.TensorShape([None, 3, 3]),
        },
        max_in_flight_samples_per_worker=1,
        flexible_batch_size=1)

    # Continue sample until we have observed all the trajectories.
    seen_lengths = set()
    while len(seen_lengths) < 10:
      sample = self._sample_from(dataset, 1)[0]

      # The structure should always be the same.
      tree.assert_same_structure(
          sample,
          replay_sample.ReplaySample(
              info=replay_sample.SampleInfo(
                  key=1,
                  probability=1.0,
                  table_size=10,
                  priority=0.5,
              ),
              data={
                  'last': None,
                  'all': None
              }))

      seen_lengths.add(sample.data['all'].shape[0])

    self.assertEqual(seen_lengths, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})


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
      trajectory_dataset.TrajectoryDataset.from_table_signature(
          address, 'not_found', 100)

  def test_server_not_found(self):
    with self.assertRaises(errors.DeadlineExceededError):
      trajectory_dataset.TrajectoryDataset.from_table_signature(
          'localhost:1234', 'not_found', 100, get_signature_timeout_secs=1)

  def test_table_does_not_have_signature(self):
    server = make_server()
    address = f'localhost:{server.port}'
    with self.assertRaisesWithPredicateMatch(
        ValueError, f'Table {TABLE} at {address} does not have a signature.'):
      trajectory_dataset.TrajectoryDataset.from_table_signature(
          address, TABLE, 100)

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

    dataset = trajectory_dataset.TrajectoryDataset.from_table_signature(
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

    dataset = trajectory_dataset.TrajectoryDataset.from_table_signature(
        f'localhost:{server.port}', 'queue', 100)
    self.assertDictEqual(
        dataset.element_spec.data, {
            'a': {
                'b': tf.TensorSpec([3, 3], tf.float32),
                'c': tf.TensorSpec([], tf.int64),
            },
        })


if __name__ == '__main__':
  tf.disable_eager_execution()
  tf.test.main()
