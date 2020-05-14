# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_grpc_library",
    "reverb_cc_library",
    "reverb_cc_proto_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
    "reverb_py_proto_library",
    "reverb_tf_deps",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_cc_test(
    name = "chunk_store_test",
    srcs = ["chunk_store_test.cc"],
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps(),
)

reverb_cc_test(
    name = "rate_limiter_test",
    srcs = ["rate_limiter_test.cc"],
    deps = [
        ":priority_table",
        "//reverb/cc/distributions:uniform",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "priority_table_test",
    srcs = ["priority_table_test.cc"],
    deps = [
        ":chunk_store",
        ":priority_table",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/distributions:fifo",
        "//reverb/cc/distributions:uniform",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "tensor_compression_test",
    srcs = ["tensor_compression_test.cc"],
    deps = [
        ":tensor_compression",
        "//reverb/cc/testing:tensor_testutil",
    ] + reverb_tf_deps(),
)

reverb_cc_test(
    name = "sampler_test",
    srcs = ["sampler_test.cc"],
    deps = [
        ":sampler",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/testing:tensor_testutil",
        "//reverb/cc/testing:time_testutil",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":client",
        ":reverb_service_cc_grpc_proto",
        ":writer",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_grpc_deps(),
)

reverb_cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":client",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_grpc_deps(),
)

reverb_cc_test(
    name = "reverb_service_impl_test",
    srcs = ["reverb_service_impl_test.cc"],
    deps = [
        ":reverb_service_cc_proto",
        ":reverb_service_impl",
        ":schema_cc_proto",
        "//reverb/cc/distributions:fifo",
        "//reverb/cc/distributions:uniform",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "chunk_store",
    srcs = ["chunk_store.cc"],
    hdrs = ["chunk_store.h"],
    deps = [
        ":schema_cc_proto",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:queue",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "priority_table",
    srcs = [
        "priority_table.cc",
        "rate_limiter.cc",
    ],
    hdrs = [
        "priority_table.h",
        "rate_limiter.h",
    ],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunk_store",
        ":priority_table_item",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/distributions:interface",
        "//reverb/cc/platform:logging",
        "//reverb/cc/table_extensions:interface",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "tensor_compression",
    srcs = ["tensor_compression.cc"],
    hdrs = ["tensor_compression.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:snappy",
    ] + reverb_tf_deps(),
)

reverb_cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "writer",
    srcs = ["writer.cc"],
    hdrs = ["writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":sampler",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":writer",
        ":schema_cc_proto",
        "//reverb/cc/platform:grpc_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:uint128",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "reverb_service_impl",
    srcs = ["reverb_service_impl.cc"],
    hdrs = ["reverb_service_impl.h"],
    deps = [
        ":chunk_store",
        ":priority_table",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:uint128",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "priority_table_item",
    hdrs = ["priority_table_item.h"],
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
    ],
)

reverb_cc_proto_library(
    name = "schema_cc_proto",
    srcs = ["schema.proto"],
)

reverb_py_proto_library(
    name = "schema_py_pb2",
    srcs = ["schema.proto"],
    deps = [":schema_cc_proto"],
)

reverb_cc_proto_library(
    name = "reverb_service_cc_proto",
    srcs = ["reverb_service.proto"],
    visibility = ["//reverb:__subpackages__"],
    deps = [":schema_cc_proto"],
)

reverb_cc_grpc_library(
    name = "reverb_service_cc_grpc_proto",
    srcs = ["reverb_service.proto"],
    generate_mocks = True,
    visibility = ["//reverb:__subpackages__"],
    deps = [":reverb_service_cc_proto"],
)
