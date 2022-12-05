load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_library",
    "reverb_cc_test",
    "reverb_gen_op_wrapper_py",
    "reverb_kernel_library",
    "reverb_tf_deps",
    "reverb_tf_ops_visibility",
)

package(default_visibility = reverb_tf_ops_visibility())

licenses(["notice"])

reverb_kernel_library(
    name = "ops",
    srcs = [
        "client.cc",
        "pattern_dataset.cc",
        "timestep_dataset.cc",
        "trajectory_dataset.cc",
    ],
    deps = [
        ":queue_writer",
        "//reverb/cc:chunker",
        "//reverb/cc:client",
        "//reverb/cc:errors",
        "//reverb/cc:sampler",
        "//reverb/cc:structured_writer",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:tf_util",
        "//reverb/cc:schema_cc_proto",
    ] + reverb_absl_deps(),
)

reverb_gen_op_wrapper_py(
    name = "gen_reverb_ops",
    out = "gen_reverb_ops.py",
    kernel_lib = ":ops",
)

reverb_cc_library(
    name = "queue_writer",
    srcs = ["queue_writer.cc"],
    hdrs = ["queue_writer.h"],
    deps = [
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:chunker",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "queue_writer_test",
    srcs = ["queue_writer_test.cc"],
    deps = [
        ":queue_writer",
        "//reverb/cc:trajectory_writer",
        "//reverb/cc:chunker",
        "//reverb/cc/support:queue",
        "//reverb/cc/platform:thread",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)
