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

"""Tests for replay_sample."""

from absl.testing import absltest

from reverb import pybind
from reverb import replay_sample


class SampleInfoTest(absltest.TestCase):

  def test_has_the_correct_number_of_fields(self):
    self.assertLen(replay_sample.SampleInfo._fields,
                   pybind.Sampler.NUM_INFO_TENSORS)


if __name__ == '__main__':
  absltest.main()
