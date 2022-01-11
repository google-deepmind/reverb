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

"""Tests for trajectory_dataset_eager."""

from absl.testing import parameterized
import numpy as np
from reverb import server as reverb_server
from reverb import trajectory_dataset
import tensorflow as tf

_TABLE = 'queue'


class TrajectoryDatasetEagerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      num_workers_per_iterator=[1, 3],
      max_in_flight_samples_per_worker=[1, 5],
      max_samples=[4],
  )
  def test_max_samples(self, num_workers_per_iterator,
                       max_in_flight_samples_per_worker, max_samples):

    server = reverb_server.Server(
        tables=[reverb_server.Table.queue(_TABLE, 10)])
    client = server.localhost_client()

    with client.trajectory_writer(10) as writer:
      for i in range(10):
        writer.append([np.ones([3, 3], np.int32) * i])
        writer.create_item(_TABLE, 1.0, {
            'last': writer.history[0][-1],
            'all': writer.history[0][:],
        })

    dataset = trajectory_dataset.TrajectoryDataset(
        tf.constant(client.server_address),
        table=tf.constant(_TABLE),
        dtypes={
            'last': tf.int32,
            'all': tf.int32,
        },
        shapes={
            'last': tf.TensorShape([3, 3]),
            'all': tf.TensorShape([None, 3, 3]),
        },
        num_workers_per_iterator=num_workers_per_iterator,
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
        max_samples=max_samples)

    # Check that it fetches exactly `max_samples` samples.
    it = dataset.as_numpy_iterator()
    self.assertLen(list(it), max_samples)
    # Check that no prefetching happen on the queue.
    self.assertEqual(client.server_info()[_TABLE].current_size,
                     10 - max_samples)


if __name__ == '__main__':
  tf.test.main()
