load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_library",
    "reverb_tf_deps",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

reverb_cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = [
        "//reverb/cc:priority_table_item",
        "//reverb/cc:schema_cc_proto",
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "base",
    srcs = ["base.cc"],
    hdrs = ["base.h"],
    deps = [
        ":interface",
        "//reverb/cc:table",
        "//reverb/cc:priority_table_item",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:logging",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)
