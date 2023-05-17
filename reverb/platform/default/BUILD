load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

py_library(
    name = "load_op_library",
    srcs = ["load_op_library.py"],
    srcs_version = "PY3",
)

py_library(
    name = "checkpointers",
    srcs = ["checkpointers.py"],
    srcs_version = "PY3",
    deps = [
        "//reverb/platform:checkpointers_lib",
    ],
)

py_library(
    name = "ensure_tf_install",
    srcs = ["ensure_tf_install.py"],
    srcs_version = "PY3",
)

py_library(
    name = "server_main_command_line_args",
    srcs = ["server_main_command_line_args.py"],
    srcs_version = "PY3",
    deps = [
        "//reverb/server_executable:reverb_config_py_pb2",
    ],
)
