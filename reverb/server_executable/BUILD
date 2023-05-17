load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_binary", "py_test")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

proto_library(
    name = "reverb_config_proto",
    srcs = ["reverb_config.proto"],
    deps = ["//reverb/cc/checkpointing:checkpoint_proto"],
)

cc_proto_library(
    name = "reverb_config_cc_proto",
    deps = [":reverb_config_proto"],
)

py_proto_library(
    name = "reverb_config_py_pb2",
    deps = [":reverb_config_proto"],
)

py_library(
    name = "server_from_proto",
    srcs = ["server_from_proto.py"],
    srcs_version = "PY3",
    deps = [
        "//reverb",
        "//reverb:reverb_types",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
    ],
)

py_test(
    name = "server_from_proto_test",
    srcs = ["server_from_proto_test.py"],
    srcs_version = "PY3",
    deps = [
        ":server_from_proto",
        "//reverb",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
    ],
)

py_binary(
    name = "server_main",
    srcs = ["server_main.py"],
    srcs_version = "PY3",
    deps = [
        ":server_from_proto",
        "//reverb",
        "//reverb:reverb_types",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
        "//reverb/platform/default:server_main_command_line_args",
    ],
    data = [
        "//reverb:pybind.so",
    ]
)
