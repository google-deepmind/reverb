load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

proto_library(
    name = "checkpoint_proto",
    srcs = ["checkpoint.proto"],
    deps = [
        "//reverb/cc:schema_proto",
        "@local_config_tf//:struct_proto",
    ],
)

cc_proto_library(
    name = "checkpoint_cc_proto",
    deps = [":checkpoint_proto"],
)

cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = ["//reverb/cc:table"]
)

py_proto_library(
    name = "checkpoint_py_pb2",
    deps = [":checkpoint_proto"],
)
