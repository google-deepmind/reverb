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


if __name__ == '__main__':
  absltest.main()
