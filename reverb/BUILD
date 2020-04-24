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
        ":distributions",
        ":errors",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":tf_client",
    ],
)

reverb_pytype_strict_library(
    name = "distributions",
    srcs = ["distributions.py"],
    srcs_version = "PY3",
    deps = [":pybind"],
)

reverb_pytype_strict_library(
    name = "rate_limiters",
    srcs = ["rate_limiters.py"],
    srcs_version = "PY3",
    deps = [":pybind"],
)

reverb_pytype_library(
    name = "client",
    srcs = ["client.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":pybind",
        ":replay_sample",
        ":reverb_types",
        "//reverb/cc:schema_py_pb2",
    ],
)

reverb_pytype_library(
    name = "errors",
    srcs = ["errors.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [],
)

reverb_pytype_library(
    name = "server",
    srcs = ["server.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":checkpointer",
        ":client",
        ":distributions",
        ":pybind",
        ":rate_limiters",
        ":reverb_types",
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
    srcs_version = "PY3ONLY",
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc:priority_table",
        "//reverb/cc:replay_client",
        "//reverb/cc:replay_writer",
        "//reverb/cc:replay_sampler",
        "//reverb/cc:reverb_server",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/distributions:fifo",
        "//reverb/cc/distributions:heap",
        "//reverb/cc/distributions:interface",
        "//reverb/cc/distributions:lifo",
        "//reverb/cc/distributions:prioritized",
        "//reverb/cc/distributions:uniform",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/table_extensions:interface",
    ] + reverb_pybind_deps() + reverb_absl_deps(),
)

reverb_pytype_library(
    name = "tf_client",
    srcs = ["tf_client.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":replay_sample",
        "//reverb/cc/ops:gen_client_ops",
        "//reverb/cc/ops:gen_dataset_op",
    ],
)

reverb_pytype_library(
    name = "checkpointer",
    srcs = ["checkpointer.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [":pybind"],
)

reverb_pytype_library(
    name = "reverb_types",
    srcs = ["reverb_types.py"],
    srcs_version = "PY3",
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":pybind",
        "//reverb/cc:schema_py_pb2",
    ],
)

reverb_py_test(
    name = "checkpointer_test",
    srcs = ["checkpointer_test.py"],
    python_version = "PY3",
    deps = [
        ":checkpointer",
        ":pybind",
    ],
)

reverb_py_test(
    name = "client_test",
    srcs = ["client_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":distributions",
        ":errors",
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
        ":distributions",
        ":rate_limiters",
        ":server",
    ],
)

reverb_py_test(
    name = "tf_client_test",
    timeout = "short",
    srcs = ["tf_client_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        ":client",
        ":distributions",
        ":rate_limiters",
        ":replay_sample",
        ":server",
        ":tf_client",
    ],
)

reverb_py_test(
    name = "rate_limiters_test",
    srcs = ["rate_limiters_test.py"],
    python_version = "PY3",
    deps = [":rate_limiters"],
)

reverb_py_test(
    name = "pybind_test",
    srcs = ["pybind_test.py"],
    python_version = "PY3",
    deps = [":reverb"],
)
