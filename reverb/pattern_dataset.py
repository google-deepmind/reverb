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

"""Dataset that transforms a dataset applying patterns."""

from typing import Any, Callable, Sequence

from reverb import replay_sample
from reverb import structured_writer
import tensorflow as tf
import tree

from reverb.cc import patterns_pb2
from reverb.cc.ops import gen_reverb_ops

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
# pylint: enable=g-direct-tensorflow-import


class PatternDataset(dataset_ops.UnaryDataset):
  """A tf.data.Dataset that applies reverb patterns to an input  dataset.

  Given a dataset (stream) of steps, this dataset will produce trajectories
  in the same way as the steps were added to a Reverb StructuredWriter and
  then sampled.

  TODO(sabela): Add examples.

  Note that this dataset can only be used in eager mode (i.e., not TF1 support).
  """

  def __init__(self,
               input_dataset: tf.data.Dataset,
               configs: Sequence[patterns_pb2.StructuredWriterConfig],
               respect_episode_boundaries: bool,
               is_end_of_episode: Callable[[Any], bool]):
    """Constructs a dataset applying the configs to the input dataset.

    Args:
      input_dataset: Dataset to apply the patterns to. It is expected to be a
        flat dataset (e.g., if you have a dataset of episodes, it has to be
        flattened into a dataset of steps).
      configs: Patterns to apply to the input dataset in order to construct
        trajectories.
      respect_episode_boundaries: If True, it won't create trajectories that
        cross episode boudaries.
      is_end_of_episode: Function that is applied to every step and returns if
        the step is the last of an episode.

    Returns:
      A dataset of trajectories.
    """
    signature = structured_writer.infer_signature(configs,
                                                  input_dataset.element_spec)
    self._shapes = tree.map_structure(lambda x: x.shape, signature)
    self._dtypes = tree.map_structure(lambda x: x.dtype, signature)
    self._input_dataset = input_dataset
    wrapped_func = structured_function.StructuredFunctionWrapper(
        is_end_of_episode,
        transformation_name='ReverbPatternDataset_IsEndOfEpisode',
        dataset=input_dataset,
        use_legacy_function=False)
    if not wrapped_func.output_structure.is_compatible_with(
        tf.TensorSpec([], tf.bool)):
      raise ValueError(
          f'Invalid `is_end_of_episode`. `is_end_of_episode` must return `bool`, '
          f'but its return type is {wrapped_func.output_structure}.')
    self._is_end_of_episode = wrapped_func

    history_length = 0
    serialized_configs = []
    for proto in configs:
      for node in proto.flat:
        history_length = max(
            history_length, abs(min(node.start, node.stop)))
        # The buffers must contain enough steps for the pattern to be applied.
        # We therefore check if the config already contains a
        # condition that ensures that the config is not applied prematurely.
        # If none of the existing conditions fulfill this responsibility then
        # we create and add one to the config.
        has_buffer_len_condition = False
        for condition in proto.conditions:
          if condition.buffer_length and condition.ge >= history_length:
            has_buffer_len_condition = True
            break
        if not has_buffer_len_condition:
          new_cond = patterns_pb2.Condition(
              buffer_length=True, ge=history_length)
          proto.conditions.append(new_cond)
      serialized_configs.append(proto.SerializeToString())
    variant_tensor = gen_reverb_ops.reverb_pattern_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._is_end_of_episode.function.captured_inputs,
        is_end_of_episode=self._is_end_of_episode.function,
        dtypes=tree.flatten(self._dtypes),
        shapes=tree.flatten(self._shapes),
        configs=serialized_configs,
        clear_after_episode=respect_episode_boundaries)
    super(PatternDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self) -> Any:
    return tree.map_structure(tf.TensorSpec, self._shapes, self._dtypes)


def pattern_dataset_with_info(
    input_dataset: tf.data.Dataset,
    configs: Sequence[patterns_pb2.StructuredWriterConfig],
    respect_episode_boundaries: bool,
    is_end_of_episode: Callable[[Any], bool]) -> tf.data.Dataset:
  """Constructs a PatternDataset and obtains the output as ReplaySamples.

  To construct the ReplaySample, it adds a SampleInfo that contains zeros.

  Args:
      input_dataset: Dataset to apply the patterns to. It is expected to be a
        flat dataset (e.g., if you have a dataset of episodes, it has to be
        flattened into a dataset of steps).
      configs: Patterns to apply to the input dataset in order to construct
        trajectories.
      respect_episode_boundaries: If True, it won't create trajectories that
        cross episode boudaries.
      is_end_of_episode: Function that is applied to every step and returns if
        the step is the last of an episode.

  Returns:
      A dataset of ReplaySamples where the info of each sample contains zeros
      and the data corresponds to the trajectories generated by applying the
      patterns.
  """

  pattern_dataset = PatternDataset(input_dataset, configs,
                                   respect_episode_boundaries,
                                   is_end_of_episode)
  @tf.function
  def build_replay_sample(data):
    return replay_sample.ReplaySample(
        info=replay_sample.SampleInfo.zeros(), data=data)

  return pattern_dataset.map(build_replay_sample)
