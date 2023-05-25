load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = [
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
    ]
)

cc_library(
    name = "uniform",
    srcs = ["uniform.cc"],
    hdrs = ["uniform.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@local_config_tf//:tf_header_lib"
    ]
)

cc_library(
    name = "fifo",
    srcs = ["fifo.cc"],
    hdrs = ["fifo.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/status",
    ]
)

cc_library(
    name = "lifo",
    srcs = ["lifo.cc"],
    hdrs = ["lifo.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/status",
    ]
)

cc_library(
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
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/random",
    ]
)

cc_library(
    name = "heap",
    srcs = ["heap.cc"],
    hdrs = ["heap.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/support:intrusive_heap",
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/status",
    ]
)

cc_test(
    name = "uniform_test",
    srcs = ["uniform_test.cc"],
    deps = [
        ":uniform",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "fifo_test",
    srcs = ["fifo_test.cc"],
    deps = [
        ":fifo",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "lifo_test",
    srcs = ["lifo_test.cc"],
    deps = [
        ":lifo",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "prioritized_test",
    srcs = ["prioritized_test.cc"],
    deps = [
        ":prioritized",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "heap_test",
    srcs = ["heap_test.cc"],
    deps = [
        ":heap",
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ],
)
