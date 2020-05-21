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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_pytype_strict_library(
    name = "reverb",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":client",
        ":distributions",
        ":ensure_tf_install",
        ":errors",
        ":item_selectors",
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
    deps = [":item_selectors"],
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
        "//reverb/cc:schema_py_pb2",
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
        ":checkpointer",
        ":client",
        ":item_selectors",
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
        "//reverb/cc:table",
        "//reverb/cc:client",
        "//reverb/cc:writer",
        "//reverb/cc:sampler",
        "//reverb/cc/platform:server",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:lifo",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/platform:checkpointing",
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
        "//reverb/cc/ops:gen_client_ops",
        "//reverb/cc/ops:gen_dataset_op",
    ],
)

reverb_pytype_library(
    name = "checkpointer",
    srcs = ["checkpointer.py"],
    srcs_version = "PY3",
    strict_deps = True,
    deps = [":pybind"],
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
    name = "ensure_tf_install",
    srcs = ["platform/default/ensure_tf_install.py"],
    srcs_version = "PY3",
)

reverb_pytype_strict_library(
    name = "item_selectors",
    srcs = ["item_selectors.py"],
    srcs_version = "PY3",
    deps = [":pybind"],
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
        ":item_selectors",
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
