load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_library",
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
