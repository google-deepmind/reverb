load(
    "//reverb/cc/platform:build_rules.bzl",
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
    name = "ops_metadata",
    srcs = [
        "client_metadata.cc",
        "pattern_dataset_metadata.cc",
        "timestep_dataset_metadata.cc",
        "trajectory_dataset_metadata.cc",
    ],
)

reverb_kernel_library(
    name = "ops",
    srcs = [
        "client.cc",
        "pattern_dataset.cc",
        "timestep_dataset.cc",
        "trajectory_dataset.cc",
    ],
    deps = [
        ":ops_metadata",
        ":queue_writer",
        "//reverb/cc:chunker",
        "//reverb/cc:client",
        "//reverb/cc:errors",
        "//reverb/cc:patterns_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:structured_writer",
        "//reverb/cc/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

reverb_gen_op_wrapper_py(
    name = "gen_reverb_ops",
    out = "gen_reverb_ops.py",
    kernel_lib = ":ops",
    ops_lib = ":ops_metadata",
)

reverb_cc_library(
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
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + reverb_tf_deps(),
)

reverb_cc_test(
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
    ] + reverb_tf_deps(),
)
