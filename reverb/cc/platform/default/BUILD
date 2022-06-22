# Platform-specific code for reverb

load(
    "//reverb/cc/platform/default:build_rules.bzl",
    "reverb_cc_library",
    "reverb_grpc_deps",
    "reverb_tf_deps",
)

package(default_visibility = ["//reverb/cc/platform:__pkg__"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_cc_library(
    name = "snappy",
    srcs = ["snappy.cc"],
    deps = [
        "//reverb/cc/platform:snappy_hdr",
        "@com_google_absl//absl/strings",
    ] + reverb_tf_deps(),
    alwayslink = 1,
)

reverb_cc_library(
    name = "checkpointer",
    srcs = ["default_checkpointer.cc"],
    deps = [
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing_hdr",
        "//reverb/cc/platform:tfrecord_checkpointer",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "checkpointing_utils",
    srcs = ["checkpointing_utils.cc"],
    deps = [
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:checkpointing_utils_hdr",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:heap",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:lifo",
        "//reverb/cc/selectors:prioritized",
        "//reverb/cc/selectors:uniform",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

reverb_cc_library(
    name = "server",
    srcs = ["server.cc"],
    deps = [
        "//reverb/cc:client",
        "//reverb/cc:reverb_service_impl",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:grpc_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:server_hdr",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:tfrecord_checkpointer",
        "//reverb/cc/support:periodic_closure",
        "@com_google_absl//absl/strings",
    ] + reverb_grpc_deps(),
    alwayslink = 1,
)

reverb_cc_library(
    name = "status_macros",
    hdrs = [
        "status_macros.h",
        "status_matchers.h",
    ],
    deps = [
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/status",
    ],
)

reverb_cc_library(
    name = "status_matchers",
    hdrs = ["status_matchers.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

reverb_cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    deps = [
        "//reverb/cc/platform:thread_hdr",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

reverb_cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = reverb_tf_deps(),
)

reverb_cc_library(
    name = "hash",
    hdrs = ["hash.h"],
    deps = reverb_tf_deps(),
)

reverb_cc_library(
    name = "hash_map",
    hdrs = ["hash_map.h"],
    deps = [
        ":hash",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

reverb_cc_library(
    name = "hash_set",
    hdrs = ["hash_set.h"],
    deps = [
        ":hash",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

reverb_cc_library(
    name = "grpc_utils",
    srcs = ["grpc_utils.cc"],
    deps = [
        "//reverb/cc/platform:grpc_utils_hdr",
    ] + reverb_grpc_deps(),
    alwayslink = 1,
)

reverb_cc_library(
    name = "net",
    srcs = ["net.cc"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:net_hdr",
    ],
    alwayslink = 1,
)
