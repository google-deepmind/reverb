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

"""Tests for reverb.trajectory_writer."""

from typing import Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reverb import client as client_lib
from reverb import pybind
from reverb import server as server_lib
from reverb import trajectory_writer
import tree


class FakeWeakCellRef:

  def __init__(self, data):
    self.data = data


def extract_data(column: trajectory_writer._ColumnHistory):
  return [ref.data if ref else None for ref in column]


class TrajectoryWriterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.cpp_writer_mock = mock.Mock()
    self.cpp_writer_mock.Append.side_effect = \
        lambda x: [FakeWeakCellRef(y) if y is not None else None for y in x]
    self.cpp_writer_mock.AppendPartial.side_effect = \
        lambda x: [FakeWeakCellRef(y) if y is not None else None for y in x]

    self.writer = trajectory_writer.TrajectoryWriter(self.cpp_writer_mock)

  def test_history_require_append_to_be_called_before(self):
    with self.assertRaises(RuntimeError):
      _ = self.writer.history

  def test_history_contains_references_when_data_flat(self):
    self.writer.append(0)
    self.writer.append(1)
    self.writer.append(2)

    history = tree.map_structure(extract_data, self.writer.history)
    self.assertListEqual(history, [0, 1, 2])

  def test_history_contains_structured_references(self):
    self.writer.append({'x': 1, 'y': 100})
    self.writer.append({'x': 2, 'y': 101})
    self.writer.append({'x': 3, 'y': 102})

    history = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(history, {'x': [1, 2, 3], 'y': [100, 101, 102]})

  def test_history_structure_evolves_with_data(self):
    self.writer.append({'x': 1, 'z': 2})
    first = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(first, {'x': [1], 'z': [2]})

    self.writer.append({'z': 3, 'y': 4})
    second = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(second, {
        'x': [1, None],
        'z': [2, 3],
        'y': [None, 4],
    })

    self.writer.append({'w': 5})
    third = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(third, {
        'x': [1, None, None],
        'z': [2, 3, None],
        'y': [None, 4, None],
        'w': [None, None, 5],
    })

    self.writer.append({'x': 6, 'w': 7})
    forth = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(forth, {
        'x': [1, None, None, 6],
        'z': [2, 3, None, None],
        'y': [None, 4, None, None],
        'w': [None, None, 5, 7],
    })

  @parameterized.named_parameters(
      ('tuple', (0,), (0, 1)),
      ('dict', {'x': 0}, {'x': 0, 'y': 1}),
      ('list', [0], [0, 1]),
  )
  def test_append_with_more_fields(self, first_step_data, second_step_data):
    self.writer.append(first_step_data)
    self.writer.append(second_step_data)

  def test_append_returns_same_structure_as_data(self):
    first_step_data = {'x': 1, 'y': 2}
    first_step_ref = self.writer.append(first_step_data)
    tree.assert_same_structure(first_step_data, first_step_ref)

    # Check that this holds true even if the data structure changes between
    # steps.
    second_step_data = {'y': 2, 'z': 3}
    second_step_ref = self.writer.append(second_step_data)
    tree.assert_same_structure(second_step_data, second_step_ref)

  def test_append_forwards_flat_data_to_cpp_writer(self):
    data = {'x': 1, 'y': 2}
    self.writer.append(data)
    self.cpp_writer_mock.Append.assert_called_with(tree.flatten(data))

  def test_partial_append_appends_to_the_same_step(self):
    # Create a first step and keep it open.
    self.writer.append({'x': 1, 'z': 2}, partial_step=True)
    first = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(first, {'x': [1], 'z': [2]})

    # Append to the same step and keep it open.
    self.writer.append({'y': 4}, partial_step=True)
    second = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(second, {
        'x': [1],
        'z': [2],
        'y': [4],
    })

    # Append to the same step and close it.
    self.writer.append({'w': 5})
    third = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(third, {
        'x': [1],
        'z': [2],
        'y': [4],
        'w': [5],
    })

    # Append to a new step.
    self.writer.append({'w': 6})
    forth = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(forth, {
        'x': [1, None],
        'z': [2, None],
        'y': [4, None],
        'w': [5, 6],
    })

  def test_columns_must_not_appear_more_than_once_in_the_same_step(self):
    # Create a first step and keep it open.
    self.writer.append({'x': 1, 'z': 2}, partial_step=True)

    # Add another unseen column alongside an existing column with a None value.
    self.writer.append({'x': None, 'y': 3}, partial_step=True)

    # Provide a value for a field that has already been set in this step.
    with self.assertRaisesRegex(
        ValueError,
        r'Field \(\'x\',\) has already been set in the active step by previous '
        r'\(partial\) append call and thus must be omitted or set to None but '
        r'got: 4'):
      self.writer.append({'x': 4})

  def test_create_item_checks_type_of_leaves(self):
    first = self.writer.append({'x': 3, 'y': 2})
    second = self.writer.append({'x': 3, 'y': 2})

    # History automatically transforms data and thus should be valid.
    self.writer.create_item('table', 1.0, {
        'x': self.writer.history['x'][0],  # Just one step.
        'y': self.writer.history['y'][:],  # Two steps.
    })

    # Columns can be constructed explicitly.
    self.writer.create_item('table', 1.0, {
        'x': trajectory_writer.TrajectoryColumn([first['x']]),
        'y': trajectory_writer.TrajectoryColumn([first['y'], second['y']])
    })

    # But all leaves must be TrajectoryColumn.
    with self.assertRaises(TypeError):
      self.writer.create_item('table', 1.0, {
          'x': trajectory_writer.TrajectoryColumn([first['x']]),
          'y': first['y'],
      })

  def test_flush_checks_block_until_num_itmes(self):
    self.writer.flush(0)
    self.writer.flush(1)
    with self.assertRaises(ValueError):
      self.writer.flush(-1)

  def test_configure_uses_auto_tune_when_max_chunk_length_not_set(self):
    self.writer.append({'x': 3, 'y': 2})
    self.writer.configure(('x',), num_keep_alive_refs=2, max_chunk_length=None)
    self.cpp_writer_mock.ConfigureChunker.assert_called_with(
        0,
        pybind.AutoTunedChunkerOptions(
            num_keep_alive_refs=2, throughput_weight=1.0))

  def test_configure_seen_column(self):
    self.writer.append({'x': 3, 'y': 2})
    self.writer.configure(('x',), num_keep_alive_refs=2, max_chunk_length=1)
    self.cpp_writer_mock.ConfigureChunker.assert_called_with(
        0,
        pybind.ConstantChunkerOptions(
            num_keep_alive_refs=2, max_chunk_length=1))

  def test_configure_unseen_column(self):
    self.writer.append({'x': 3, 'y': 2})
    self.writer.configure(('z',), num_keep_alive_refs=2, max_chunk_length=1)

    # The configure call should be delayed until the column has been observed.
    self.cpp_writer_mock.ConfigureChunker.assert_not_called()

    # Still not seen.
    self.writer.append({'a': 4})
    self.cpp_writer_mock.ConfigureChunker.assert_not_called()

    self.writer.append({'z': 5})
    self.cpp_writer_mock.ConfigureChunker.assert_called_with(
        3,
        pybind.ConstantChunkerOptions(
            num_keep_alive_refs=2, max_chunk_length=1))

  @parameterized.parameters(
      (1, None, True),
      (0, None, False),
      (-1, None, False),
      (1, 1, True),
      (1, 0, False),
      (1, -1, False),
      (5, 5, True),
      (4, 5, False),
  )
  def test_configure_validates_params(self, num_keep_alive_refs: int,
                                      max_chunk_length: Optional[int],
                                      valid: bool):
    if valid:
      self.writer.configure(('a',),
                            num_keep_alive_refs=num_keep_alive_refs,
                            max_chunk_length=max_chunk_length)
    else:
      with self.assertRaises(ValueError):
        self.writer.configure(('a',),
                              num_keep_alive_refs=num_keep_alive_refs,
                              max_chunk_length=max_chunk_length)

  def test_episode_steps(self):
    for _ in range(10):
      # Every episode, including the first, should start at zero.
      self.assertEqual(self.writer.episode_steps, 0)

      for i in range(1, 21):
        self.writer.append({'x': 3, 'y': 2})

        # Step count should increment with each append call.
        self.assertEqual(self.writer.episode_steps, i)

      # Ending the episode should reset the step count to zero.
      self.writer.end_episode()


class TrajectoryColumnTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = server_lib.Server([server_lib.Table.queue('queue', 100)])

  def setUp(self):
    super().setUp()
    self.client = client_lib.Client(f'localhost:{self._server.port}')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def test_numpy(self):
    writer = self.client.trajectory_writer(num_keep_alive_refs=10)

    for i in range(10):
      writer.append({'a': i, 'b': np.ones([3, 3], np.float) * i})

      np.testing.assert_array_equal(writer.history['a'][:].numpy(),
                                    np.arange(i + 1, dtype=np.int64))

      np.testing.assert_array_equal(
          writer.history['b'][:].numpy(),
          np.stack([np.ones([3, 3], np.float) * x for x in range(i + 1)]))

  def test_numpy_squeeze(self):
    writer = self.client.trajectory_writer(num_keep_alive_refs=10)

    for i in range(10):
      writer.append({'a': i})
      self.assertEqual(writer.history['a'][-1].numpy(), i)

  def test_validates_squeeze(self):
    # Exactly one is valid.
    trajectory_writer.TrajectoryColumn([FakeWeakCellRef(1)], squeeze=True)

    # Zero is not fine.
    with self.assertRaises(ValueError):
      trajectory_writer.TrajectoryColumn([], squeeze=True)

    # Neither is two (or more).
    with self.assertRaises(ValueError):
      trajectory_writer.TrajectoryColumn(
          [FakeWeakCellRef(1), FakeWeakCellRef(2)], squeeze=True)

  def test_len(self):
    for i in range(1, 10):
      column = trajectory_writer.TrajectoryColumn([FakeWeakCellRef(1)] * i)
      self.assertLen(column, i)

  def test_none_raises(self):
    with self.assertRaisesRegex(ValueError, r'cannot contain any None'):
      trajectory_writer.TrajectoryColumn([None])

    with self.assertRaisesRegex(ValueError, r'cannot contain any None'):
      trajectory_writer.TrajectoryColumn([FakeWeakCellRef(1), None])

if __name__ == '__main__':
  absltest.main()
