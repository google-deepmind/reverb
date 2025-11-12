load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_tf_deps",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

reverb_cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = [
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "uniform",
    srcs = ["uniform.cc"],
    hdrs = ["uniform.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "fifo",
    srcs = ["fifo.cc"],
    hdrs = ["fifo.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "lifo",
    srcs = ["lifo.cc"],
    hdrs = ["lifo.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "prioritized",
    srcs = ["prioritized.cc"],
    hdrs = ["prioritized.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "heap",
    srcs = ["heap.cc"],
    hdrs = ["heap.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:intrusive_heap",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "uniform_test",
    srcs = ["uniform_test.cc"],
    deps = [
        ":interface",
        ":uniform",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "fifo_test",
    srcs = ["fifo_test.cc"],
    deps = [
        ":fifo",
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "lifo_test",
    srcs = ["lifo_test.cc"],
    deps = [
        ":interface",
        ":lifo",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "prioritized_test",
    srcs = ["prioritized_test.cc"],
    deps = [
        ":interface",
        ":prioritized",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "heap_test",
    srcs = ["heap_test.cc"],
    deps = [
        ":heap",
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
        "//third_party/absl/status",
    ],
)
