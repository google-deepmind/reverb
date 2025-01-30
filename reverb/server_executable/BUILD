load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_cc_proto_library",
    "reverb_py_proto_library",
    "reverb_py_test",
    "reverb_pytype_strict_binary",
    "reverb_pytype_strict_library",
)

package(
    default_visibility = ["//reverb:__subpackages__"],
)

licenses(["notice"])

reverb_cc_proto_library(
    name = "reverb_config_cc_proto",
    srcs = ["reverb_config.proto"],
    deps = ["//reverb/cc/checkpointing:checkpoint_cc_proto"],
)

reverb_py_proto_library(
    name = "reverb_config_py_pb2",
    srcs = ["reverb_config.proto"],
    deps = [
        ":reverb_config_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
    ],
)

reverb_pytype_strict_library(
    name = "server_from_proto",
    srcs = ["server_from_proto.py"],
    deps = [
        "//reverb",
        "//reverb:reverb_types",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
    ],
)

reverb_py_test(
    name = "server_from_proto_test",
    srcs = ["server_from_proto_test.py"],
    enable_dashboard = False,
    deps = [
        ":server_from_proto",
        "//reverb",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
    ],
)

reverb_pytype_strict_binary(
    name = "server_main",
    srcs = ["server_main.py"],
    deps = [
        ":server_from_proto",
        "//reverb",
        "//reverb:pybind",
        "//reverb:reverb_types",
        "//reverb/cc:schema_py_pb2",
        "//reverb/cc/checkpointing:checkpoint_py_pb2",
        "//reverb/platform/default:server_main_command_line_args",
    ],
)
