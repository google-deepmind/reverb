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

from unittest import mock

from absl.testing import absltest
from reverb import trajectory_writer
import tree


class FakeWeakCellRef:

  def __init__(self, data):
    self.data = data


def extract_data(column: trajectory_writer._ColumnHistory):
  return [ref.data if ref else None for ref in column[:]]


class TrajectoryWriterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.cpp_writer_mock = mock.Mock()
    self.cpp_writer_mock.Append.side_effect = \
        lambda x: [FakeWeakCellRef(y) if y else None for y in x]

    client_mock = mock.Mock()
    client_mock._client.NewTrajectoryWriter.return_value = self.cpp_writer_mock

    self.writer = trajectory_writer.TrajectoryWriter(client_mock, 1, 1)

  def test_history_require_append_to_be_called_before(self):
    with self.assertRaises(RuntimeError):
      _ = self.writer.history

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

    self.writer.append({'y': 5})
    third = tree.map_structure(extract_data, self.writer.history)
    self.assertDictEqual(third, {
        'x': [1, None, None],
        'z': [2, 3, None],
        'y': [None, 4, 5],
    })

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

  def test_configure_seen_column(self):
    self.writer.append({'x': 3, 'y': 2})
    self.writer.configure(('x',), 1, 2)
    self.cpp_writer_mock.ConfigureChunker.assert_called_with(0, 1, 2)

  def test_configure_unseen_column(self):
    self.writer.append({'x': 3, 'y': 2})
    self.writer.configure(('z',), 1, 2)

    # The configure call should be delayed until the column has been observed.
    self.cpp_writer_mock.ConfigureChunker.assert_not_called()

    # Still not seen.
    self.writer.append({'a': 4})
    self.cpp_writer_mock.ConfigureChunker.assert_not_called()

    self.writer.append({'z': 5})
    self.cpp_writer_mock.ConfigureChunker.assert_called_with(3, 1, 2)


if __name__ == '__main__':
  absltest.main()
