# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

config_setting(
    name = "macos",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

py_library(
    name = "reverb",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":client",
        ":errors",
        ":item_selectors",
        ":pattern_dataset",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":structured_writer",
        ":tf_client",
        ":timestep_dataset",
        ":trajectory_dataset",
        ":trajectory_writer",
        "//reverb/platform/default:checkpointers",
        "//reverb/platform/default:ensure_tf_install",
    ],
)

pybind_extension(
    name = "pybind",
    srcs = ["pybind.cc"],
    copts = ["-pthread"],
    visibility = [":__subpackages__"],
    deps = [
        "//reverb/cc:chunker",
        "//reverb/cc:client",
        "//reverb/cc:conversions",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:structured_writer",
        "//reverb/cc:table",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc:writer",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:checkpointing_utils",
        "//reverb/cc/platform:server",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:lifo",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/table_extensions:interface",
        "@com_google_absl//absl/random",
    ],
)

py_library(
    name = "rate_limiters",
    srcs = ["rate_limiters.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
)

py_library(
    name = "client",
    srcs = ["client.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":replay_sample",
        ":reverb_types",
        ":structured_writer",
        ":trajectory_writer",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "server",
    srcs = ["server.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
    deps = [
        ":client",
        ":item_selectors",
        ":rate_limiters",
        ":reverb_types",
        "//reverb/platform/default:checkpointers",
    ],
)

py_library(
    name = "replay_sample",
    srcs = ["replay_sample.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tf_client",
    srcs = ["tf_client.py"],
    data = ["//reverb/cc/ops:gen_reverb_ops"],
    srcs_version = "PY3",
    deps = [
        ":replay_sample",
    ],
)

py_library(
    name = "trajectory_dataset",
    srcs = ["trajectory_dataset.py"],
    data = ["//reverb/cc/ops:gen_reverb_ops"],
    srcs_version = "PY3",
    visibility = [":__subpackages__"],
    deps = [
        ":client",
        ":replay_sample",
    ],
)

py_library(
    name = "timestep_dataset",
    srcs = ["timestep_dataset.py"],
    data = ["//reverb/cc/ops:gen_reverb_ops"],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":client",
        ":replay_sample",
    ],
)

py_library(
    name = "reverb_types",
    srcs = ["reverb_types.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
    deps = ["//reverb/cc:schema_py_pb2"],
)

py_library(
    name = "structured_writer",
    srcs = ["structured_writer.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":reverb_types",
        "//reverb/cc:patterns_py_pb2",
    ],
)

py_library(
    name = "item_selectors",
    srcs = ["item_selectors.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
)

py_library(
    name = "trajectory_writer",
    srcs = ["trajectory_writer.py"],
    data = [":pybind.so"],
    srcs_version = "PY3",
    deps = [":errors"],
)

py_test(
    name = "client_test",
    srcs = ["client_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":client",
        ":errors",
        ":item_selectors",
        ":rate_limiters",
        ":server",
    ],
)

py_test(
    name = "server_test",
    srcs = ["server_test.py"],
    data = [":pybind.so"],
    python_version = "PY3",
    deps = [
        ":client",
        ":item_selectors",
        ":rate_limiters",
        ":server",
    ],
)

py_test(
    name = "tf_client_test",
    timeout = "short",
    srcs = ["tf_client_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":item_selectors",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":tf_client",
    ],
)

py_test(
    name = "timestep_dataset_test",
    srcs = ["timestep_dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":errors",
        ":item_selectors",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":timestep_dataset",
    ],
)

py_test(
    name = "timestep_dataset_eager_test",
    srcs = ["timestep_dataset_eager_test.py"],
    deps = [
        ":server",
        ":timestep_dataset",
    ],
)

py_test(
    name = "trajectory_dataset_test",
    srcs = ["trajectory_dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":errors",
        ":item_selectors",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":trajectory_dataset",
    ],
)

py_test(
    name = "trajectory_dataset_eager_test",
    srcs = ["trajectory_dataset_eager_test.py"],
    deps = [
        ":server",
        ":trajectory_dataset",
    ],
)

py_test(
    name = "rate_limiters_test",
    srcs = ["rate_limiters_test.py"],
    python_version = "PY3",
    deps = [
        ":rate_limiters",
    ],
)

py_test(
    name = "pybind_test",
    srcs = ["pybind_test.py"],
    python_version = "PY3",
    deps = [
        ":reverb",
    ],
)

py_test(
    name = "trajectory_writer_test",
    srcs = ["trajectory_writer_test.py"],
    data = [":pybind.so"],
    python_version = "PY3",
    deps = [
        ":client",
        ":errors",
        ":server",
        ":trajectory_writer",
    ],
)

py_test(
    name = "structured_writer_test",
    srcs = ["structured_writer_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":server",
        ":structured_writer",
    ],
)

py_test(
    name = "replay_sample_test",
    srcs = ["replay_sample_test.py"],
    data = [":pybind.so"],
    python_version = "PY3",
    deps = [
        ":replay_sample",
    ],
)

py_library(
    name = "pattern_dataset",
    srcs = ["pattern_dataset.py"],
    srcs_version = "PY3",
    tags = ["pytype_unchecked_annotations"],
    deps = [
        ":replay_sample",
        ":structured_writer",
        "//reverb/cc:patterns_py_pb2",
        "//reverb/cc/ops:gen_reverb_ops",
    ],
)

py_test(
    name = "pattern_dataset_test",
    srcs = ["pattern_dataset_test.py"],
    deps = [
        ":pattern_dataset",
        ":replay_sample",
        ":structured_writer",
    ],
)
