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

"""Define Reverb version information."""

# We follow Semantic Versioning (https://semver.org/)
load(
    "@tf_wheel_version_suffix//:wheel_version_suffix.bzl",
    "WHEEL_VERSION_SUFFIX",
)

REVERB_VERSION = "0.15.0"
MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = REVERB_VERSION.split(".")

REVERB_VERSION_SUFFIX = WHEEL_VERSION_SUFFIX
REVERB_TENSORFLOW_RELEASE_VERSION = "tensorflow~=2.21.0"
REVERB_TENSORFLOW_NIGHTLY_VERSION = "tf_nightly~=2.21.0.dev"
