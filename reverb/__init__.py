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

"""Top level import for Reverb."""

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
from reverb.platform.default import ensure_tf_install

ensure_tf_install.ensure_tf_version()

# Cleanup symbols to avoid polluting namespace.
del ensure_tf_install
# pylint: enable=g-bad-import-order

from reverb import item_selectors as selectors
from reverb import rate_limiters

from reverb import structured_writer as structured

from reverb.client import Client
from reverb.client import Writer

from reverb.errors import DeadlineExceededError
from reverb.errors import ReverbError

from reverb.pattern_dataset import PatternDataset

from reverb.platform.default import checkpointers

from reverb.replay_sample import ReplaySample
from reverb.replay_sample import SampleInfo

from reverb.server import Server
from reverb.server import Table

from reverb.tf_client import TFClient

from reverb.timestep_dataset import TimestepDataset

from reverb.trajectory_dataset import TrajectoryDataset

from reverb.trajectory_writer import TrajectoryColumn
from reverb.trajectory_writer import TrajectoryWriter
