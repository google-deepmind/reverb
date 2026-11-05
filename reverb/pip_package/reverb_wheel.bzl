# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Collect Reverb artifacts and build a wheel with the target Python ABI."""

load("@reverb_wheel_config//:config.bzl", "WHEEL_NAME")
load("@rules_python//python:py_info.bzl", "PyInfo")
load(
    ":reverb_version.bzl",
    "REVERB_TENSORFLOW_NIGHTLY_VERSION",
    "REVERB_TENSORFLOW_RELEASE_VERSION",
    "REVERB_VERSION",
    "REVERB_VERSION_SUFFIX",
)

def _reverb_wheel_impl(ctx):
    runtime = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    version_info = runtime.interpreter_version_info
    python_tag = "cp%s%s" % (version_info.major, version_info.minor)
    platform = ctx.attr.platform
    version = REVERB_VERSION + REVERB_VERSION_SUFFIX
    filename = "%s-%s-%s-%s-%s.whl" % (WHEEL_NAME, version, python_tag, python_tag, platform)
    output = ctx.actions.declare_file("wheel_house/" + filename)
    files = depset(
        direct = ctx.files.source_files + [
            f
            for dep in ctx.attr.deps
            for f in dep[DefaultInfo].default_runfiles.files.to_list()
            if f.extension in ["so", "dylib", "pyd"]
        ],
        transitive = [dep[PyInfo].transitive_sources for dep in ctx.attr.deps],
    ).to_list()
    files = [f for f in files if f.owner.repo_name == ctx.label.repo_name]
    destinations = {}
    for f in files:
        path = f.short_path
        if path.startswith("../"):
            path = "/".join(path.split("/")[2:])
        destinations[f.path] = path
    manifest = ctx.actions.declare_file(ctx.label.name + "_sources.json")
    ctx.actions.write(manifest, json.encode(destinations))
    args = ctx.actions.args()
    args.add("--project-name", WHEEL_NAME)
    args.add("--platform", platform)
    args.add("--python-tag", python_tag)
    args.add("--output-name", output.dirname)
    args.add("--version", version)
    args.add("--tf-version", REVERB_TENSORFLOW_NIGHTLY_VERSION if "nightly" in WHEEL_NAME else REVERB_TENSORFLOW_RELEASE_VERSION)
    args.add("--dests", manifest)
    ctx.actions.run(
        arguments = [args],
        inputs = files + [manifest],
        outputs = [output],
        executable = ctx.attr.wheel_binary[DefaultInfo].files_to_run,
        mnemonic = "BuildWheel",
    )
    return [DefaultInfo(files = depset([output]))]

reverb_wheel = rule(
    implementation = _reverb_wheel_impl,
    attrs = {
        "source_files": attr.label_list(allow_files = True),
        "deps": attr.label_list(providers = [PyInfo]),
        "platform": attr.string(mandatory = True),
        "wheel_binary": attr.label(
            default = Label("//reverb/pip_package:build_wheel"),
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = ["@rules_python//python:toolchain_type"],
)
