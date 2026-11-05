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

"""Sanity tests for the pybind.py."""

import threading

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import reverb

TABLE_NAME = 'queue'


class TestNdArrayToTensorAndBack(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestNdArrayToTensorAndBack, cls).setUpClass()
    cls._server = reverb.Server(tables=[reverb.Table.queue(TABLE_NAME, 1000)])
    cls._client = cls._server.localhost_client()

  def tearDown(self):
    super(TestNdArrayToTensorAndBack, self).tearDown()
    self._client.reset(TABLE_NAME)

  @classmethod
  def tearDownClass(cls):
    super(TestNdArrayToTensorAndBack, cls).tearDownClass()
    cls._server.stop()

  @parameterized.parameters(
      (1,),
      (1.0,),
      (np.arange(4).reshape([2, 2]),),
      (np.array(1, dtype=np.float16),),
      (np.array(1, dtype=np.float32),),
      (np.array(1, dtype=np.float64),),
      (np.array(1, dtype=np.int8),),
      (np.array(1, dtype=np.int16),),
      (np.array(1, dtype=np.int32),),
      (np.array(1, dtype=np.int64),),
      (np.array(1, dtype=np.uint8),),
      (np.array(1, dtype=np.uint16),),
      (np.array(1, dtype=np.uint32),),
      (np.array(1, dtype=np.uint64),),
      (np.array(True, dtype=bool),),
      (np.array(1, dtype=np.complex64),),
      (np.array(1, dtype=np.complex128),),
      (np.array([b'a string']),),
  )
  def test_sanity_check(self, data):
    with self._client.writer(1) as writer:
      writer.append([data])
      writer.create_item(TABLE_NAME, 1, 1)

    sample = next(self._client.sample(TABLE_NAME))
    got = sample[0].data[0]
    np.testing.assert_array_equal(data, got)

  def test_stress_string_memory_leak(self):
    with self._client.writer(1) as writer:
      for i in range(100):
        writer.append(['string_' + ('a' * 100 * i)])
        writer.create_item(TABLE_NAME, 1, 1)

    for i in range(100):
      sample = next(self._client.sample(TABLE_NAME))
      got = sample[0].data[0]
      np.testing.assert_array_equal(got, b'string_' + (b'a' * 100 * i))


class TestSamplerControls(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = reverb.Server(tables=[reverb.Table.queue(TABLE_NAME, 100)])
    self.client = self.server.localhost_client()
    self.addCleanup(self.server.stop)

  def sampler(self, *args, **kwargs):
    sampler = self.client._client.NewSampler(*args, **kwargs)
    self.addCleanup(sampler.Close)
    return sampler

  def test_positional_defaults(self):
    self.client.insert(np.float32(3), {TABLE_NAME: 1.0})
    sampler = self.sampler(TABLE_NAME, 1, 1)
    np.testing.assert_array_equal(sampler.GetNextTrajectory()[5], [3])

  @parameterized.parameters(1, 2)
  def test_worker_count_and_keyword_arguments(self, num_workers):
    for value in range(8):
      self.client.insert(np.float32(value), {TABLE_NAME: 1.0})
    sampler = self.sampler(
        table=TABLE_NAME, max_samples=8, buffer_size=2,
        num_workers=num_workers, rate_limiter_timeout_ms=10000)
    values = [int(sampler.GetNextTrajectory()[5][0]) for _ in range(8)]
    self.assertCountEqual(values, range(8))
    if num_workers == 1:
      self.assertEqual(values, list(range(8)))

  @parameterized.parameters(0, -2)
  def test_invalid_worker_count(self, num_workers):
    with self.assertRaisesRegex(ValueError, 'num_workers'):
      self.sampler(TABLE_NAME, 1, 1, num_workers=num_workers)

  def test_invalid_timeout(self):
    with self.assertRaisesRegex(ValueError, 'rate_limiter_timeout'):
      self.sampler(TABLE_NAME, 1, 1, rate_limiter_timeout_ms=-2)

  @parameterized.parameters(0, 20)
  def test_timeout_exception(self, timeout_ms):
    sampler = self.sampler(
        TABLE_NAME, 1, 1, rate_limiter_timeout_ms=timeout_ms)
    with self.assertRaises(reverb.DeadlineExceededError):
      sampler.GetNextTrajectory()

  def test_close_cancels_blocked_read(self):
    sampler = self.sampler(TABLE_NAME, -1, 1)
    entered = threading.Event()
    finished = threading.Event()
    outcomes = []

    def read():
      entered.set()
      try:
        outcomes.append(sampler.GetNextTrajectory())
      except Exception as error:  # pylint: disable=broad-except
        outcomes.append(error)
      finally:
        finished.set()

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
      self.assertTrue(entered.wait(timeout=10))
      self.assertFalse(finished.wait(timeout=0.05))
    finally:
      sampler.Close()
      thread.join(timeout=10)
    self.assertFalse(thread.is_alive())
    self.assertLen(outcomes, 1)
    self.assertIsInstance(outcomes[0], RuntimeError)
    self.assertIn('cancelled', str(outcomes[0]).lower())
    sampler.Close()


if __name__ == '__main__':
  absltest.main()
