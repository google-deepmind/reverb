load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

cc_library(
    name = "uint128",
    hdrs = ["uint128.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
    ],
)

cc_library(
    name = "grpc_util",
    hdrs = ["grpc_util.h"],
)

cc_library(
    name = "queue",
    hdrs = ["queue.h"],
    deps = [
        "//reverb/cc/platform:logging",
    ],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
)

cc_test(
    name = "cleanup_test",
    srcs = ["cleanup_test.cc"],
    deps = [
        ":cleanup",
    ],
)

cc_library(
    name = "intrusive_heap",
    srcs = ["intrusive_heap.cc"],
    hdrs = ["intrusive_heap.h"],
    deps = [
        "//reverb/cc/platform:logging",
    ],
)

cc_test(
    name = "intrusive_heap_test",
    srcs = ["intrusive_heap_test.cc"],
    deps = [
        ":intrusive_heap",
    ],
)

cc_library(
    name = "periodic_closure",
    srcs = ["periodic_closure.cc"],
    hdrs = ["periodic_closure.h"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "queue_test",
    srcs = ["queue_test.cc"],
    deps = [
        ":queue",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
    ],
)

cc_test(
    name = "periodic_closure_test",
    srcs = ["periodic_closure_test.cc"],
    deps = [
        ":periodic_closure",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:time_testutil",
    ],
)

cc_library(
    name = "signature",
    srcs = ["signature.cc"],
    hdrs = ["signature.h"],
    deps = [
        ":trajectory_util",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
    ],
)

cc_test(
    name = "signature_test",
    srcs = ["signature_test.cc"],
    deps = [
        ":signature",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
    ],
)

cc_library(
    name = "state_statistics",
    hdrs = [
        "state_statistics.h",
    ],
    deps = [
        "//reverb/cc/platform:hash_map",
    ],
)

cc_test(
    name = "state_statistics_test",
    srcs = ["state_statistics_test.cc"],
    deps = [
        ":state_statistics",
    ],
)

cc_library(
    name = "task_executor",
    srcs = ["task_executor.cc"],
    hdrs = ["task_executor.h"],
    deps = [
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:periodic_closure",
        "//reverb/cc/support:unbounded_queue",
    ],
)

cc_library(
    name = "tf_util",
    hdrs = ["tf_util.h"],
    deps = [
    ],
)

cc_library(
    name = "unbounded_queue",
    hdrs = ["unbounded_queue.h"],
)

cc_test(
    name = "unbounded_queue_test",
    srcs = ["unbounded_queue_test.cc"],
    deps = [
        ":unbounded_queue",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
    ],
)

cc_library(
    name = "trajectory_util",
    srcs = ["trajectory_util.cc"],
    hdrs = ["trajectory_util.h"],
    deps = [
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
    ],
)

cc_test(
    name = "trajectory_util_test",
    srcs = ["trajectory_util_test.cc"],
    deps = [
        ":trajectory_util",
        "//reverb/cc:chunk_store",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ],
)

cc_library(
    name = "key_generators",
    hdrs = ["key_generators.h"],
)
