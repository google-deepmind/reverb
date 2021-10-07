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

"""Get server configuration (constructor args) from textproto cli arg.
"""
from absl import flags
from reverb.server_executable import reverb_config_pb2

from google.protobuf import text_format


_CONFIG = flags.DEFINE_string(
    "config", None, "Reverb server config in textproto format",
    required=True)


def get_server_config_proto() -> reverb_config_pb2.ReverbServerConfig:
  config_proto = reverb_config_pb2.ReverbServerConfig()
  text_format.Parse(_CONFIG.value, config_proto)
  return config_proto
