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

"""Tests for structured_writer."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reverb import client as client_lib
from reverb import server as server_lib
from reverb import structured_writer
import tree

Config = structured_writer.Config
Node = structured_writer.Node
Condition = structured_writer.Condition
ModuloEq = structured_writer.ModuloEq


class _RefNode:

  def __init__(self, flat_source_index: int):
    self._flat_source_index = flat_source_index

  def __getitem__(self, key):
    if isinstance(key, int):
      key = slice(key)

    return Node(
        flat_source_index=self._flat_source_index,
        start=key.start,
        stop=key.stop,
        step=key.step)


TABLES = tuple(f'queue_{i}' for i in range(5))

STEP_SPEC = {
    'a': np.zeros([], np.float32),
    'b': {
        'c': np.zeros([2, 2], np.int32),
        'd': [
            np.zeros([3], np.int64),
            np.zeros([6], np.int64),
        ],
    },
}

REF_STEP = tree.unflatten_as(
    STEP_SPEC, [_RefNode(i) for i in range(len(tree.flatten(STEP_SPEC)))])


def create_step(idx, structure):
  return tree.map_structure(lambda x: np.ones_like(x) * idx, structure)


class StructuredWriterTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = server_lib.Server(
        [server_lib.Table.queue(table, 100) for table in TABLES])

  def setUp(self):
    super().setUp()
    self.client = client_lib.Client(f'localhost:{self._server.port}')

  def tearDown(self):
    super().tearDown()
    for table in TABLES:
      self.client.reset(table)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def get_table_content(self, idx: int, structure=None):
    info = self.client.server_info(1)
    num_items = info[TABLES[idx]].current_size
    sampler = self.client.sample(
        TABLES[idx], num_samples=num_items, emit_timesteps=False)

    flat_samples = [sample.data for sample in sampler]

    if structure:
      return [tree.unflatten_as(structure, sample) for sample in flat_samples]

    return flat_samples

  @parameterized.parameters(
      {
          'condition': Condition(step_index=True, le=2),
          'num_steps': 10,
          'want_steps': [0, 1, 2],
      },
      {
          'condition': Condition(step_index=True, ge=5),
          'num_steps': 10,
          'want_steps': [5, 6, 7, 8, 9],
      },
      {
          'condition': Condition(step_index=True, eq=3),
          'num_steps': 10,
          'want_steps': [3],
      },
      {
          'condition': Condition(step_index=True, mod_eq=ModuloEq(mod=3, eq=0)),
          'num_steps': 10,
          'want_steps': [0, 3, 6, 9],
      },
      {
          'condition': Condition(steps_since_applied=True, ge=4),
          'num_steps': 10,
          'want_steps': [3, 7],
      },
      {
          'condition': Condition(steps_since_applied=True, ge=3),
          'num_steps': 10,
          'want_steps': [2, 5, 8],
      },
      {
          'condition': Condition(is_end_episode=True, eq=1),
          'num_steps': 5,
          'want_steps': [4],
      },
  )
  def test_single_condition(self, condition, num_steps, want_steps):
    config = Config(
        flat=[Node(flat_source_index=0, stop=-1)],
        table=TABLES[0],
        priority=1.0,
        conditions=[condition],
    )

    writer = self.client.structured_writer([config])

    for i in range(num_steps):
      writer.append(i)

    writer.end_episode()

    want = [[step] for step in want_steps]
    self.assertEqual(self.get_table_content(0), want)

  @parameterized.parameters(
      {
          'pattern': {
              'x': REF_STEP['a'][-3],
              'y': REF_STEP['b']['c'][-2:],
              'z': REF_STEP['b']['d'][0][-1],
          },
          'num_steps':
              5,
          'want': [
              {
                  'x':
                      np.array(0, np.float32),
                  'y':
                      np.array([
                          np.array([[1, 1], [1, 1]], np.int32),
                          np.array([[2, 2], [2, 2]], np.int32),
                      ]),
                  'z':
                      np.array([2, 2, 2], np.int64),
              },
              {
                  'x':
                      np.array(1, np.float32),
                  'y':
                      np.stack([
                          np.array([[2, 2], [2, 2]], np.int32),
                          np.array([[3, 3], [3, 3]], np.int32),
                      ]),
                  'z':
                      np.array([3, 3, 3], np.int64),
              },
              {
                  'x':
                      np.array(2, np.float32),
                  'y':
                      np.stack([
                          np.array([[3, 3], [3, 3]], np.int32),
                          np.array([[4, 4], [4, 4]], np.int32),
                      ]),
                  'z':
                      np.array([4, 4, 4], np.int64),
              },
          ],
      },
      {
          'pattern': {
              'older': REF_STEP['a'][-3:-1],
              'last_a': REF_STEP['a'][-1],
          },
          'num_steps':
              5,
          'want': [
              {
                  'older': np.array([0, 1], np.float32),
                  'last_a': np.array(2, np.float32),
              },
              {
                  'older': np.array([1, 2], np.float32),
                  'last_a': np.array(3, np.float32),
              },
              {
                  'older': np.array([2, 3], np.float32),
                  'last_a': np.array(4, np.float32),
              },
          ],
      },
      {
          'pattern': {
              'every_second_a': REF_STEP['a'][-5::2]
          },
          'num_steps':
              10,
          'want': [
              {
                  'every_second_a': np.array([0, 2, 4], np.float32)
              },
              {
                  'every_second_a': np.array([1, 3, 5], np.float32)
              },
              {
                  'every_second_a': np.array([2, 4, 6], np.float32)
              },
              {
                  'every_second_a': np.array([3, 5, 7], np.float32)
              },
              {
                  'every_second_a': np.array([4, 6, 8], np.float32)
              },
              {
                  'every_second_a': np.array([5, 7, 9], np.float32)
              },
          ],
      },
  )
  def test_trajectory_patterns(self, pattern, num_steps, want):
    config = Config(
        flat=tree.flatten(pattern),
        table=TABLES[0],
        priority=1.0,
    )

    writer = self.client.structured_writer([config])

    for i in range(num_steps):
      writer.append(create_step(i, STEP_SPEC))

    writer.end_episode()

    tree.map_structure(np.testing.assert_array_equal, want,
                       self.get_table_content(0, pattern))

  def test_round_robin_into_tables(self):
    # Create configs which should result in steps being written to available
    # tables in a round robin fashion.
    configs = []
    for i, table in enumerate(TABLES):
      configs.append(
          Config(
              flat=[Node(flat_source_index=0, stop=-1)],
              table=table,
              priority=1.0,
              conditions=[
                  Condition(
                      step_index=True,
                      mod_eq=ModuloEq(mod=len(TABLES), eq=i),
                  ),
              ],
          ))

    # Take enough steps to generate two trajectories for each table.
    writer = self.client.structured_writer(configs)
    for i in range(len(TABLES) * 2):
      writer.append(i)
    writer.end_episode()

    # Check that steps was inserted into tables in the expected order.
    self.assertEqual(self.get_table_content(0), [[0], [5]])
    self.assertEqual(self.get_table_content(1), [[1], [6]])
    self.assertEqual(self.get_table_content(2), [[2], [7]])
    self.assertEqual(self.get_table_content(3), [[3], [8]])
    self.assertEqual(self.get_table_content(4), [[4], [9]])


if __name__ == '__main__':
  absltest.main()
