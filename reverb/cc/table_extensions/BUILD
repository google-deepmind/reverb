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
        "//reverb/cc:schema_cc_proto",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_library(
    name = "base",
    srcs = ["base.cc"],
    hdrs = ["base.h"],
    deps = [
        ":interface",
        "//reverb/cc:table",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:logging",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_library(
    name = "async_base",
    srcs = ["async_base.cc"],
    hdrs = ["async_base.h"],
    deps = [
        ":interface",
        "//reverb/cc:table",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_test(
    name = "async_base_test",
    srcs = ["async_base_test.cc"],
    deps = [
        ":async_base",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)
