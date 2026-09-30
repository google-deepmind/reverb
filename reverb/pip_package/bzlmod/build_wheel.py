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

# Copyright 2023 The Tensorflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stage declared Reverb artifacts and build a wheel without network access."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def prepare_srcs(manifest: str, destination: str) -> None:
  """Copy manifest entries relative to the wheel root."""
  with open(manifest, encoding="utf-8") as stream:
    sources = json.load(stream)
  for source, relative in sources.items():
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
      raise ValueError(f"Invalid wheel destination: {relative!r}")
    target = Path(destination) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o644)


def main() -> None:
  parser = argparse.ArgumentParser()
  for name in (
      "output-name", "project-name", "platform", "python-tag", "version",
      "tf-version", "dests",
  ):
    parser.add_argument("--" + name, required=True)
  args = parser.parse_args()
  output = os.path.abspath(args.output_name)
  with tempfile.TemporaryDirectory(prefix="reverb_wheel") as staging:
    prepare_srcs(args.dests, staging)
    for name in ("pyproject.toml", "hatch_build.py"):
      shutil.move(
          os.path.join(staging, "reverb", "pip_package", "bzlmod", name),
          os.path.join(staging, name),
      )
    env = dict(
        os.environ,
        project_name=args.project_name,
        version=args.version,
        tf_version=args.tf_version,
        plat_name=args.platform,
        python_tag=args.python_tag,
    )
    # Pass the launcher's dependency paths to the build-backend subprocess.
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run(
        [
            sys.executable, "-m", "build", "--wheel", "--no-isolation",
            "--outdir", output,
        ],
        check=True,
        cwd=staging,
        env=env,
    )


if __name__ == "__main__":
  main()
