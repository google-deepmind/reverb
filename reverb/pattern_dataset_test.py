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

"""Tests for writer_dataset."""

import numpy as np
from reverb import pattern_dataset
from reverb import replay_sample
from reverb import structured_writer
import tensorflow as tf


# Many things we use in RLDS & tf.data do not support TF1
class PatternDatasetTest(tf.test.TestCase):

  def test_apply_pattern_and_condition(self):
    pattern = structured_writer.pattern_from_transform(
        step_structure=None, transform=lambda x: x[-1])
    condition = structured_writer.Condition.step_index() <= 2
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table', conditions=[condition])

    input_dataset = tf.data.Dataset.from_tensor_slices({
        'observation': {
            'field_0': [0, 1, 2, 3],
            'field_1': [0, 1, 2, 3],
        },
        'action': [0, 1, 2, 3],
        'is_last': [False, False, False, False],
    })
    expected_dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2])

    output_dataset = pattern_dataset.PatternDataset(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: x['is_last'])

    num_elements = 0
    for out_step, expected_step in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertAllClose(out_step, expected_step)
      num_elements += 1

    self.assertEqual(num_elements, 3)

  def test_apply_pattern_keeps_tensor_shape(self):

    input_dataset = tf.data.Dataset.from_tensor_slices([[0, 0], [1, 1], [2, 2],
                                                        [3, 3]])
    expected_dataset = tf.data.Dataset.from_tensor_slices([[0, 0], [1, 1],
                                                           [2, 2]])
    ref_step = structured_writer.create_reference_step(
        input_dataset.element_spec)
    print(input_dataset.element_spec)

    pattern = structured_writer.pattern_from_transform(
        step_structure=ref_step, transform=lambda x: x[-1])
    condition = structured_writer.Condition.step_index() <= 2
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table', conditions=[condition])

    output_dataset = pattern_dataset.PatternDataset(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: False)

    num_elements = 0
    for out_step, expected_step in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertAllClose(out_step, expected_step)
      num_elements += 1

    self.assertEqual(num_elements, 3)

  def test_apply_pattern_keeps_batch_dimension(self):

    input_dataset = tf.data.Dataset.from_tensor_slices([[0, 0], [1, 1], [2, 2],
                                                        [3, 3]])
    expected_dataset = tf.data.Dataset.from_tensor_slices([[[0, 0], [1, 1]],
                                                           [[1, 1], [2, 2]],
                                                           [[2, 2], [3, 3]]])
    ref_step = structured_writer.create_reference_step(
        input_dataset.element_spec)

    pattern = structured_writer.pattern_from_transform(
        step_structure=ref_step, transform=lambda x: x[-2:])
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table')

    output_dataset = pattern_dataset.PatternDataset(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: False)

    num_elements = 0
    for out_step, expected_step in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertAllClose(out_step, expected_step)
      num_elements += 1

    self.assertEqual(num_elements, 3)

  def test_apply_pattern_nested_result(self):
    step_spec = {
        'observation': {
            'field_0': np.zeros([], np.int64),
            'field_1': np.zeros([], np.int64),
        },
        'action': np.zeros([], np.int64),
        'is_last': np.zeros([], bool),
    }
    ref_step = structured_writer.create_reference_step(step_spec)

    pattern = {
        'observation': {
            'field_0': ref_step['observation']['field_0'][-2],
            'field_1': ref_step['observation']['field_1'][-2],
        },
        'next_observation': {
            'field_0': ref_step['observation']['field_0'][-1],
            'field_1': ref_step['observation']['field_1'][-1],
        },
    }
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table')

    input_dataset = tf.data.Dataset.from_tensor_slices({
        'observation': {
            'field_0': [0, 1, 2, 3],
            'field_1': [0, 1, 2, 3],
        },
        'action': [0, 1, 2, 3],
        'is_last': [False, False, False, False],
    })

    expected_dataset = tf.data.Dataset.from_tensor_slices({
        'observation': {
            'field_0': [0, 1, 2],
            'field_1': [0, 1, 2],
        },
        'next_observation': {
            'field_0': [1, 2, 3],
            'field_1': [1, 2, 3],
        },
    })

    output_dataset = pattern_dataset.PatternDataset(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: x['is_last'])

    num_elements = 0
    for out_step, expected_step in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertAllClose(out_step, expected_step)
      num_elements += 1

    self.assertEqual(num_elements, 3)

  def test_respects_end_of_episode(self):

    step_spec = {
        'data': np.zeros([], np.int64),
        'is_last': np.zeros([], bool),
    }
    ref_step = structured_writer.create_reference_step(step_spec)

    pattern = {'data': ref_step['data'][-2], 'next': ref_step['data'][-1]}
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table')

    input_dataset = tf.data.Dataset.from_tensor_slices({
        'data': [0, 1, 2, 3, 4, 5, 6],
        'is_last': [False, False, False, True, False, False, True],
    })
    # There are no pairs created that span between two episodes
    expected_dataset = tf.data.Dataset.from_tensor_slices({
        'data': [0, 1, 2, 4, 5],
        'next': [1, 2, 3, 5, 6],
    })

    output_dataset = pattern_dataset.PatternDataset(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: x['is_last'])

    num_elements = 0
    for out_step, expected_step in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertAllClose(out_step, expected_step)
      num_elements += 1

    self.assertEqual(num_elements, 5)

  def test_ignores_end_of_episode(self):

    step_spec = {
        'data': np.zeros([], np.int64),
        'is_last': np.zeros([], bool),
    }
    ref_step = structured_writer.create_reference_step(step_spec)

    pattern = {'data': ref_step['data'][-2], 'next': ref_step['data'][-1]}
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table')

    input_dataset = tf.data.Dataset.from_tensor_slices({
        'data': [0, 1, 2, 3, 4, 5, 6],
        'is_last': [False, False, False, True, False, False, True],
    })
    # There is one pair with data in the previous episode, and next in the
    # next one.
    expected_dataset = tf.data.Dataset.from_tensor_slices({
        'data': [0, 1, 2, 3, 4, 5],
        'next': [1, 2, 3, 4, 5, 6],
    })

    output_dataset = pattern_dataset.PatternDataset(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=False,
        is_end_of_episode=lambda x: x['is_last'])

    num_elements = 0
    for out_step, expected_step in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertAllClose(out_step, expected_step)
      num_elements += 1

    self.assertEqual(num_elements, 6)

  def test_build_replay_sample_adds_sample_info(self):
    pattern = structured_writer.pattern_from_transform(
        step_structure=None, transform=lambda x: x[-1])
    condition = structured_writer.Condition.step_index() <= 2
    config = structured_writer.create_config(
        pattern=pattern, table='fake_table', conditions=[condition])

    input_dataset = tf.data.Dataset.from_tensor_slices({'data': [0, 1, 2, 3]})
    expected_dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2])

    output_dataset = pattern_dataset.pattern_dataset_with_info(
        input_dataset=input_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda _: False)

    num_elements = 0
    for step, expected_data in tf.data.Dataset.zip(
        (output_dataset, expected_dataset)):
      self.assertEqual(step.info, replay_sample.SampleInfo.zeros())
      self.assertEqual(step.data, expected_data)
      num_elements += 1

    self.assertEqual(num_elements, 3)


if __name__ == '__main__':
  tf.test.main()
