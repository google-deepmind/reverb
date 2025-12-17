load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

reverb_cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

reverb_cc_library(
    name = "base",
    srcs = ["base.cc"],
    hdrs = ["base.h"],
    deps = [
        ":interface",
        "//reverb/cc:table",
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)
