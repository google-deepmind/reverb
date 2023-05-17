load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
    ]# + reverb_absl_deps()
)

cc_library(
    name = "base",
    srcs = ["base.cc"],
    hdrs = ["base.h"],
    deps = [
        ":interface",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/platform:logging",
    ]
)
