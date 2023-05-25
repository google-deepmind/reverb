load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
package(default_visibility = ["//reverb:__subpackages__"])
licenses(["notice"])

py_library(
    name = "gen_reverb_ops",
    srcs = [
        "__init__.py",
        "gen_reverb_ops.py",
    ],
    data = [
        ":libgen_reverb_ops_gen_op.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libgen_reverb_ops_gen_op.so",
    srcs = [
        "client.cc",
        "pattern_dataset.cc",
        "timestep_dataset.cc",
        "trajectory_dataset.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":queue_writer",
        "//reverb/cc:chunker",
        "//reverb/cc:client",
        "//reverb/cc:errors",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:structured_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:tf_util",
        "@local_config_tf//:tf_header_lib",
        "@local_config_tf//:libtensorflow_framework",
    ],
    linkshared = True,
    copts = [
        "-pthread",
        "-std=c++17",
        "-undefined dynamic_lookup",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ],
)

cc_library(
    name = "queue_writer",
    srcs = ["queue_writer.cc"],
    hdrs = ["queue_writer.h"],
    deps = [
        "//reverb/cc:chunker",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
    ],
)

cc_test(
    name = "queue_writer_test",
    srcs = ["queue_writer_test.cc"],
    deps = [
        ":queue_writer",
        "//reverb/cc:chunker",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:queue",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ],
)
