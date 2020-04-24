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

"""Tests for Reverb rate limiters."""

from absl.testing import absltest
from absl.testing import parameterized
from reverb import rate_limiters


class TestSampleToInsertRatio(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'less_than_samples_per_insert',
          'samples_per_insert': 5,
          'error_buffer': 4,
          'want': ValueError,
      },
      {
          'testcase_name': 'less_than_one',
          'samples_per_insert': 0.5,
          'error_buffer': 0.9,
          'want': ValueError,
      },
      {
          'testcase_name': 'valid',
          'samples_per_insert': 0.5,
          'error_buffer': 1.1,
          'want': None,
      },
  )
  def test_validates_single_number_error_buffer(self, samples_per_insert,
                                                error_buffer, want):
    if want:
      with self.assertRaises(want):
        rate_limiters.SampleToInsertRatio(samples_per_insert, 10, error_buffer)
    else:  # Should not raise any error.
      rate_limiters.SampleToInsertRatio(samples_per_insert, 10, error_buffer)

  @parameterized.named_parameters(
      {
          'testcase_name': 'range_too_small_due_to_sample_per_insert_ratio',
          'min_size_to_sample': 10,
          'samples_per_insert': 5,
          'error_buffer': (8, 12),
          'want': ValueError,
      },
      {
          'testcase_name': 'range_smaller_than_2',
          'min_size_to_sample': 10,
          'samples_per_insert': 0.1,
          'error_buffer': (9.5, 10.5),
          'want': ValueError,
      },
      {
          'testcase_name': 'range_below_min_size_to_sample',
          'min_size_to_sample': 10,
          'samples_per_insert': 1,
          'error_buffer': (5, 9),
          'want': None,
      },
      {
          'testcase_name': 'range_above_min_size_to_sample',
          'min_size_to_sample': 10,
          'samples_per_insert': 1,
          'error_buffer': (11, 15),
          'want': ValueError,
      },
      {
          'testcase_name': 'min_size_to_sample_smaller_than_1',
          'min_size_to_sample': 0,
          'samples_per_insert': 1,
          'error_buffer': (-100, 100),
          'want': ValueError,
      },
      {
          'testcase_name': 'valid',
          'min_size_to_sample': 10,
          'samples_per_insert': 1,
          'error_buffer': (7, 12),
          'want': None,
      },
  )
  def test_validates_explicit_range_error_buffer(self, min_size_to_sample,
                                                 samples_per_insert,
                                                 error_buffer, want):
    if want:
      with self.assertRaises(want):
        rate_limiters.SampleToInsertRatio(samples_per_insert,
                                          min_size_to_sample, error_buffer)
    else:  # Should not raise any error.
      rate_limiters.SampleToInsertRatio(samples_per_insert, min_size_to_sample,
                                        error_buffer)


class TestMinSize(parameterized.TestCase):

  @parameterized.parameters(
      (-1, True),
      (0, True),
      (1, False),
  )
  def test_raises_if_min_size_lt_1(self, min_size_to_sample, want_error):
    if want_error:
      with self.assertRaises(ValueError):
        rate_limiters.MinSize(min_size_to_sample)
    else:
      rate_limiters.MinSize(min_size_to_sample)


if __name__ == '__main__':
  absltest.main()
