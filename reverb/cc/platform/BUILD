# Platform-specific code for reverb

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
    "reverb_tf_deps",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

reverb_cc_library(
    name = "tfrecord_checkpointer",
    srcs = ["tfrecord_checkpointer.cc"],
    hdrs = ["tfrecord_checkpointer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":checkpointing_utils",
        ":hash_map",
        ":hash_set",
        ":logging",
        ":status_macros",
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/table_extensions:interface",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "tfrecord_checkpointer_test",
    srcs = ["tfrecord_checkpointer_test.cc"],
    deps = [
        ":status_matchers",
        ":tfrecord_checkpointer",
        "//reverb/cc:chunk_store",
        "//reverb/cc:table",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/table_extensions:base",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps(),
)

reverb_cc_library(
    name = "checkpointing_hdr",
    hdrs = ["checkpointing.h"],
    deps = [
        "//reverb/cc/checkpointing:interface",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "checkpointing",
    hdrs = ["checkpointing.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform/default:checkpointer",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "checkpointing_utils_hdr",
    hdrs = ["checkpointing_utils.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/selectors:interface",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "checkpointing_utils",
    hdrs = ["checkpointing_utils.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:checkpointing_utils",
        "//reverb/cc/selectors:interface",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "grpc_utils_hdr",
    hdrs = ["grpc_utils.h"],
    deps = reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "grpc_utils",
    hdrs = ["grpc_utils.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:grpc_utils",
    ] + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "server_hdr",
    hdrs = ["server.h"],
    deps = [
        "//reverb/cc:client",
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:interface",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "server",
    hdrs = ["server.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc:client",
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform/default:server",
    ] + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "server_test",
    srcs = ["server_test.cc"],
    deps = [
        ":server",
        "//reverb/cc/platform:net",
        "//reverb/cc/platform:status_matchers",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "snappy_hdr",
    hdrs = ["snappy.h"],
    deps = reverb_absl_deps(),
)

reverb_cc_library(
    name = "snappy",
    hdrs = ["snappy.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:snappy",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:status_macros"],
)

reverb_cc_library(
    name = "status_matchers",
    hdrs = ["status_matchers.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:status_matchers"],
)

reverb_cc_library(
    name = "thread_hdr",
    hdrs = ["thread.h"],
    deps = reverb_absl_deps(),
)

reverb_cc_library(
    name = "thread",
    hdrs = ["thread.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform/default:thread",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "thread_test",
    srcs = ["thread_test.cc"],
    deps = [
        ":thread",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "hash_map",
    hdrs = ["hash_map.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:hash_map"],
)

reverb_cc_library(
    name = "hash_set",
    hdrs = ["hash_set.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:hash_set"],
)

reverb_cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:logging"],
)

reverb_cc_library(
    name = "net_hdr",
    hdrs = ["net.h"],
)

reverb_cc_library(
    name = "net",
    hdrs = ["net.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = ["//reverb/cc/platform/default:net"],
)

reverb_cc_test(
    name = "net_test",
    srcs = ["net_test.cc"],
    deps = [
        ":logging",
        ":net",
    ],
)
