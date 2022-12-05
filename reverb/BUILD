# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_py_test",
    "reverb_pybind_deps",
    "reverb_pybind_extension",
    "reverb_pytype_library",
    "reverb_pytype_strict_library",
)

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_pytype_strict_library(
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

reverb_pytype_strict_library(
    name = "rate_limiters",
    srcs = ["rate_limiters.py"],
    srcs_version = "PY3",
    deps = [
        ":pybind",
    ],
)

reverb_pytype_library(
    name = "client",
    srcs = ["client.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [
        ":errors",
        ":pybind",
        ":replay_sample",
        ":reverb_types",
        ":structured_writer",
        ":trajectory_writer",
    ],
)

reverb_pytype_library(
    name = "errors",
    srcs = ["errors.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [],
)

reverb_pytype_library(
    name = "server",
    srcs = ["server.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [
        ":client",
        ":item_selectors",
        ":pybind",
        ":rate_limiters",
        ":reverb_types",
        "//reverb/platform/default:checkpointers",
    ],
)

reverb_pytype_library(
    name = "replay_sample",
    srcs = ["replay_sample.py"],
    srcs_version = "PY3",
    strict_deps = True,
)

reverb_pybind_extension(
    name = "pybind",
    srcs = ["pybind.cc"],
    module_name = "libpybind",
    pytype_srcs = ["pybind.pyi"],
    srcs_version = "PY3ONLY",
    visibility = [":__subpackages__"],
    deps = [
        "//reverb/cc:chunker",
        "//reverb/cc/support:tf_util",
        "//reverb/cc:client",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:structured_writer",
        "//reverb/cc:table",
        "//reverb/cc:conversions",
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
        "//reverb/cc/table_extensions:interface",
    ] + reverb_pybind_deps() + reverb_absl_deps(),
)

reverb_pytype_library(
    name = "tf_client",
    srcs = ["tf_client.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [
        ":replay_sample",
        "//reverb/cc/ops:gen_reverb_ops",
    ],
)

reverb_pytype_library(
    name = "trajectory_dataset",
    srcs = ["trajectory_dataset.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = [":__subpackages__"],
    deps = [
        ":client",
        ":replay_sample",
        "//reverb/cc/ops:gen_reverb_ops",
    ],
)

reverb_pytype_library(
    name = "timestep_dataset",
    srcs = ["timestep_dataset.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:private"],
    deps = [
        ":client",
        ":replay_sample",
        "//reverb/cc/ops:gen_reverb_ops",
    ],
)

reverb_pytype_library(
    name = "reverb_types",
    srcs = ["reverb_types.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [
        ":pybind",
        "//reverb/cc:schema_py_pb2",
    ],
)

reverb_pytype_strict_library(
    name = "structured_writer",
    srcs = ["structured_writer.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":pybind",
        ":reverb_types",
        "//reverb/cc:patterns_py_pb2",
    ],
)

reverb_pytype_strict_library(
    name = "item_selectors",
    srcs = ["item_selectors.py"],
    srcs_version = "PY3",
    deps = [":pybind"],
)

reverb_pytype_library(
    name = "trajectory_writer",
    srcs = ["trajectory_writer.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [
        ":errors",
        ":pybind",
    ],
)

reverb_py_test(
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

reverb_py_test(
    name = "server_test",
    srcs = ["server_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":item_selectors",
        ":pybind",
        ":rate_limiters",
        ":server",
    ],
)

reverb_py_test(
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

reverb_py_test(
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

reverb_py_test(
    name = "timestep_dataset_eager_test",
    srcs = ["timestep_dataset_eager_test.py"],
    deps = [
        ":server",
        ":timestep_dataset",
    ],
)

reverb_py_test(
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

reverb_py_test(
    name = "trajectory_dataset_eager_test",
    srcs = ["trajectory_dataset_eager_test.py"],
    deps = [
        ":server",
        ":trajectory_dataset",
    ],
)

reverb_py_test(
    name = "rate_limiters_test",
    srcs = ["rate_limiters_test.py"],
    python_version = "PY3",
    deps = [
        ":rate_limiters",
    ],
)

reverb_py_test(
    name = "pybind_test",
    srcs = ["pybind_test.py"],
    python_version = "PY3",
    deps = [
        ":reverb",
    ],
)

reverb_py_test(
    name = "trajectory_writer_test",
    srcs = ["trajectory_writer_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":errors",
        ":pybind",
        ":server",
        ":trajectory_writer",
    ],
)

reverb_py_test(
    name = "structured_writer_test",
    srcs = ["structured_writer_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":server",
        ":structured_writer",
    ],
)

reverb_py_test(
    name = "replay_sample_test",
    srcs = ["replay_sample_test.py"],
    python_version = "PY3",
    deps = [
        ":pybind",
        ":replay_sample",
    ],
)

reverb_pytype_library(
    name = "pattern_dataset",
    srcs = ["pattern_dataset.py"],
    srcs_version = "PY3",
    strict_deps = False,
    tags = ["pytype_unchecked_annotations"],
    deps = [
        ":replay_sample",
        ":structured_writer",
        "//reverb/cc:patterns_py_pb2",
        "//reverb/cc/ops:gen_reverb_ops",
    ],
)

reverb_py_test(
    name = "pattern_dataset_test",
    srcs = ["pattern_dataset_test.py"],
    deps = [
        ":pattern_dataset",
        ":replay_sample",
        ":structured_writer",
    ],
)
