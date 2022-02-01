load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_gen_op_wrapper_py",
    "reverb_kernel_library",
    "reverb_tf_ops_visibility",
)

package(default_visibility = reverb_tf_ops_visibility())

licenses(["notice"])

reverb_kernel_library(
    name = "ops",
    srcs = [
        "client.cc",
        "timestep_dataset.cc",
        "trajectory_dataset.cc",
    ],
    deps = [
        "//reverb/cc:client",
        "//reverb/cc:errors",
        "//reverb/cc:sampler",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:tf_util",
    ] + reverb_absl_deps(),
)

reverb_gen_op_wrapper_py(
    name = "gen_reverb_ops",
    out = "gen_reverb_ops.py",
    kernel_lib = ":ops",
)
