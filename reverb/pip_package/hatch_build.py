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

# pylint: disable=g-importing-member
"""Custom build hook for Reverb.

See
  https://hatch.pypa.io/latest/how-to/config/dynamic-metadata/
for more details.
"""

import os
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomBuildHook(BuildHookInterface):

  def initialize(self, version: str, build_data: dict[str, Any]):
    build_data['infer_tag'] = True
    build_data['platform'] = os.environ['plat_name']


class MetaDataHook(MetadataHookInterface):

  def update(self, metadata: dict[str, Any]) -> None:
    metadata['version'] = os.environ['version']
    tf_version = os.environ['tf_version']
    metadata['optional-dependencies'] = {'tensorflow': [tf_version]}
    metadata['name'] = os.environ['project_name']
