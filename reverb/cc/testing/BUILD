load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_tf_deps",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

reverb_cc_library(
    name = "proto_test_util",
    testonly = 1,
    srcs = ["proto_test_util.cc"],
    hdrs = ["proto_test_util.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:tensor_compression",
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/log:check",
    ] + reverb_tf_deps(),
)

reverb_cc_library(
    name = "tensor_testutil",
    testonly = 1,
    hdrs = ["tensor_testutil.h"],
    deps = [
        "//reverb/cc/platform:logging",
    ] + reverb_tf_deps(),
)

reverb_cc_library(
    name = "time_testutil",
    testonly = 1,
    hdrs = ["time_testutil.h"],
    deps = ["@com_google_absl//absl/time"],
)
