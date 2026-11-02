"""Tests for wheel staging across Bazel repositories."""

import json
from pathlib import Path
import tempfile
import unittest

import build_wheel


class WheelStagingTest(unittest.TestCase):

  def test_external_generated_artifact_keeps_package_path(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "bazel-out" / "bin" / "external" / "reverb+" / "schema_pb2.py"
      source.parent.mkdir(parents=True)
      source.write_text("generated schema")
      manifest = root / "manifest.json"
      manifest.write_text(json.dumps({str(source): "reverb/cc/schema_pb2.py"}))
      build_wheel.prepare_srcs(str(manifest), str(root / "staging"))
      self.assertEqual(
          (root / "staging/reverb/cc/schema_pb2.py").read_text(),
          "generated schema",
      )

  def test_destination_cannot_escape_wheel(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      manifest = root / "manifest.json"
      for destination in ("../outside.py", "/absolute.py"):
        manifest.write_text(json.dumps({"unused": destination}))
        with self.assertRaises(ValueError):
          build_wheel.prepare_srcs(str(manifest), str(root / "staging"))


if __name__ == "__main__":
  unittest.main()
