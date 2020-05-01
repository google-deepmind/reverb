load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
    "reverb_tf_deps",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

reverb_cc_library(
    name = "uint128",
    hdrs = ["uint128.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "grpc_util",
    hdrs = ["grpc_util.h"],
    deps = reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "queue",
    hdrs = ["queue.h"],
    deps = reverb_absl_deps(),
)

reverb_cc_library(
    name = "intrusive_heap",
    srcs = ["intrusive_heap.cc"],
    hdrs = ["intrusive_heap.h"],
    deps = [
        "//reverb/cc/platform:logging",
    ],
)

reverb_cc_test(
    name = "intrusive_heap_test",
    srcs = ["intrusive_heap_test.cc"],
    deps = [
        ":intrusive_heap",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "periodic_closure",
    srcs = ["periodic_closure.cc"],
    hdrs = ["periodic_closure.h"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_test(
    name = "queue_test",
    srcs = ["queue_test.cc"],
    deps = [
        ":queue",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "periodic_closure_test",
    srcs = ["periodic_closure_test.cc"],
    deps = [
        ":periodic_closure",
        "//reverb/cc/testing:time_testutil",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_library(
    name = "signature",
    srcs = ["signature.cc"],
    hdrs = ["signature.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)
