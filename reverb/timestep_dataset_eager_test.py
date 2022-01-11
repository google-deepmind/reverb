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

"""Tests for timestep_dataset_eager."""

from absl.testing import parameterized
import numpy as np
from reverb import server as reverb_server
from reverb import timestep_dataset
import tensorflow as tf


class TimestepDatasetEagerTest(parameterized.TestCase):

  @parameterized.product(
      num_workers_per_iterator=[1, 3],
      max_in_flight_samples_per_worker=[1, 5],
      max_samples=[4],
  )
  def test_max_samples(self, num_workers_per_iterator,
                       max_in_flight_samples_per_worker, max_samples):
    s = reverb_server.Server([reverb_server.Table.queue('q', 10)])
    c = s.localhost_client()

    for i in range(10):
      c.insert(i, {'q': 1})

    ds = timestep_dataset.TimestepDataset(
        server_address=c.server_address,
        table='q',
        dtypes=tf.int64,
        shapes=tf.TensorShape([]),
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
        num_workers_per_iterator=num_workers_per_iterator,
        max_samples=max_samples)

    # Check that it fetches exactly `max_samples` samples.
    it = ds.as_numpy_iterator()
    self.assertLen(list(it), max_samples)

    # Check that no prefetching happened in the queue.
    self.assertEqual(c.server_info()['q'].current_size, 10 - max_samples)
    np.testing.assert_array_equal(
        next(c.sample('q', 1))[0].data[0], np.asarray(max_samples))


if __name__ == '__main__':
  tf.test.main()
