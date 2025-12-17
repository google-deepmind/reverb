load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_library",
    "reverb_cc_proto_library",
    "reverb_py_proto_library",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

reverb_cc_proto_library(
    name = "checkpoint_cc_proto",
    srcs = ["checkpoint.proto"],
    deps = [
        "//reverb/cc:schema_cc_proto",
    ],
)

reverb_cc_library(
    name = "interface",
    hdrs = ["interface.h"],
    deps = [
        "//reverb/cc:table",
        "@com_google_absl//absl/status",
    ],
)

reverb_py_proto_library(
    name = "checkpoint_py_pb2",
    srcs = ["checkpoint.proto"],
    deps = [
        ":checkpoint_cc_proto",
        "//reverb/cc:schema_cc_proto",
    ],
)
