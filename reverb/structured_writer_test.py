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

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import tensor_spec
# pylint: enable=g-direct-tensorflow-import

Condition = structured_writer.Condition
TensorSpec = tensor_spec.TensorSpec

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

REF_STEP = structured_writer.create_reference_step(STEP_SPEC)


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
          'condition': Condition.step_index() <= 2,
          'num_steps': 10,
          'want_steps': [0, 1, 2],
      },
      {
          'condition': Condition.step_index() >= 5,
          'num_steps': 10,
          'want_steps': [5, 6, 7, 8, 9],
      },
      {
          'condition': Condition.step_index() == 3,
          'num_steps': 10,
          'want_steps': [3],
      },
      {
          'condition': Condition.step_index() != 3,
          'num_steps': 10,
          'want_steps': [0, 1, 2, 4, 5, 6, 7, 8, 9],
      },
      {
          'condition': Condition.step_index() % 3 == 0,
          'num_steps': 10,
          'want_steps': [0, 3, 6, 9],
      },
      {
          'condition': Condition.step_index() % 3 != 0,
          'num_steps': 10,
          'want_steps': [1, 2, 4, 5, 7, 8],
      },
      {
          'condition': Condition.steps_since_applied() >= 4,
          'num_steps': 10,
          'want_steps': [3, 7],
      },
      {
          'condition': Condition.steps_since_applied() >= 3,
          'num_steps': 10,
          'want_steps': [2, 5, 8],
      },
      {
          'condition': Condition.is_end_episode(),
          'num_steps': 5,
          'want_steps': [4],
      },
  )
  def test_single_condition(self, condition, num_steps, want_steps):
    pattern = structured_writer.pattern_from_transform(
        step_structure=None, transform=lambda x: x[-1])

    config = structured_writer.create_config(
        pattern=pattern, table=TABLES[0], conditions=[condition])

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
    config = structured_writer.create_config(
        pattern=pattern, table=TABLES[0], conditions=[])

    writer = self.client.structured_writer([config])

    for i in range(num_steps):
      writer.append(create_step(i, STEP_SPEC))

    writer.end_episode()

    tree.map_structure(np.testing.assert_array_equal, want,
                       self.get_table_content(0, pattern))

  def test_round_robin_into_tables(self):
    pattern = structured_writer.pattern_from_transform(
        step_structure=None, transform=lambda x: x[-1])

    # Create configs which should result in steps being written to available
    # tables in a round robin fashion.
    configs = []
    for i, table in enumerate(TABLES):
      configs.append(
          structured_writer.create_config(
              pattern=pattern,
              table=table,
              conditions=[Condition.step_index() % len(TABLES) == i]))

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


class TestInferSignature(parameterized.TestCase):

  @parameterized.parameters(
      {
          'patterns': [{
              'older': REF_STEP['a'][-3:-1],
              'last_a': REF_STEP['a'][-1],
          },],
          'step_spec': {
              'a': np.zeros([3, 3], np.float32),
          },
          'want': {
              'older': TensorSpec([2, 3, 3], np.float32, 'older'),
              'last_a': TensorSpec([3, 3], np.float32, 'last_a'),
          },
      },
      {
          'patterns': [{
              'a_with_step': REF_STEP['a'][-6::2],
              'a_slice': REF_STEP['a'][-4:],
              'x': {
                  'y': REF_STEP['b']['c'][-2],
              },
          },],
          'step_spec': {
              'a': np.zeros([3, 3], np.float32),
              'b': {
                  'c': np.zeros([], np.int32),
                  'd': np.zeros([5], np.int8),  # Unused.
              },
          },
          'want': {
              'a_with_step': TensorSpec([3, 3, 3], np.float32, 'a_with_step'),
              'a_slice': TensorSpec([4, 3, 3], np.float32, 'a_slice'),
              'x': {
                  'y': TensorSpec([], np.int32, 'x/y'),
              },
          },
      },
      {
          'patterns': [
              {
                  'x': REF_STEP['a'][-3:-1],
                  'y': REF_STEP['a'][-1],
                  'z': REF_STEP['b']['c'][-4:],
              },
              {
                  'x': REF_STEP['a'][-2:],
                  'y': REF_STEP['a'][-2],
                  'z': REF_STEP['b']['c'][-8::2],
              },
          ],
          'step_spec': {
              'a': np.zeros([3, 3], np.float32),
              'b': {
                  'c': np.zeros([2, 2], np.int8),
              },
          },
          'want': {
              'x': TensorSpec([2, 3, 3], np.float32, 'x'),
              'y': TensorSpec([3, 3], np.float32, 'y'),
              'z': TensorSpec([4, 2, 2], np.int8, 'z'),
          },
      },
      {
          'patterns': [
              {
                  'x': REF_STEP['a'][-3:-1],
              },
              {
                  'x': REF_STEP['a'][-2:],
              },
              {
                  'x': REF_STEP['a'][-3:],
              },
          ],
          'step_spec': {
              'a': np.zeros([3, 3], np.float32),
          },
          'want': {
              'x': TensorSpec([None, 3, 3], np.float32, 'x'),
          },
      },
  )
  def test_valid_configs(self, patterns, step_spec, want):
    configs = [
        structured_writer.create_config(pattern, 'table')
        for pattern in patterns
    ]
    got = structured_writer.infer_signature(configs, step_spec)
    self.assertEqual(want, got)

  def test_requires_same_table(self):
    pattern = {'x': REF_STEP['a'][-3:]}
    configs = [
        structured_writer.create_config(pattern, 'a'),
        structured_writer.create_config(pattern, 'b'),
        structured_writer.create_config(pattern, 'c'),
    ]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'All configs must target the same table but provided configs included '
        'a, b, c.'):
      structured_writer.infer_signature(configs, STEP_SPEC)

  def test_requires_same_pattern_structure(self):
    configs = [
        structured_writer.create_config({'x': REF_STEP['a'][-1]}, 'a'),
        structured_writer.create_config({'y': REF_STEP['a'][-1]}, 'a'),
    ]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'All configs must have exactly the same pattern_structure.'):
      structured_writer.infer_signature(configs, STEP_SPEC)

  def test_requires_at_least_one_config(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'At least one config must be provided.'):
      structured_writer.infer_signature([], STEP_SPEC)

  def test_requires_same_dtype(self):
    step_spec = {
        'a': np.zeros([], np.float32),
        'b': np.zeros([], np.float64),
    }
    ref_step = structured_writer.create_reference_step(step_spec)
    configs = [
        structured_writer.create_config({'x': ref_step['a'][-1]}, 'a'),
        structured_writer.create_config({'x': ref_step['b'][-1]}, 'a'),
    ]
    with self.assertRaisesRegex(
        ValueError,
        r'Configs produce trajectories with multiple dtypes at \(\'x\',\)\. '
        r'Got .*'):
      structured_writer.infer_signature(configs, step_spec)

  def test_requires_same_rank(self):
    step_spec = {
        'a': np.zeros([], np.float32),
        'b': np.zeros([1], np.float32),
    }
    ref_step = structured_writer.create_reference_step(step_spec)
    configs = [
        structured_writer.create_config({'x': ref_step['a'][-1]}, 'a'),
        structured_writer.create_config({'x': ref_step['b'][-1]}, 'a'),
    ]
    with self.assertRaisesRegex(
        ValueError, r'Configs produce trajectories with incompatible shapes at '
        r'\(\'x\',\)\. Got .*'):
      structured_writer.infer_signature(configs, step_spec)

  def test_requires_same_concatable_shapes(self):
    step_spec = {
        'a': np.zeros([1, 2], np.float32),
        'b': np.zeros([1, 3], np.float32),
    }
    ref_step = structured_writer.create_reference_step(step_spec)
    configs = [
        structured_writer.create_config({'x': ref_step['a'][-1]}, 'a'),
        structured_writer.create_config({'x': ref_step['b'][-1]}, 'a'),
    ]
    with self.assertRaisesRegex(
        ValueError, r'Configs produce trajectories with incompatible shapes at '
        r'\(\'x\',\)\. Got .*'):
      structured_writer.infer_signature(configs, step_spec)


if __name__ == '__main__':
  absltest.main()
