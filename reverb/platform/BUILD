load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_py_test",
    "reverb_pytype_strict_library",
)

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

reverb_pytype_strict_library(
    name = "checkpointers_lib",
    srcs = ["checkpointers_lib.py"],
    deps = [
        "//reverb:pybind",
    ],
)

reverb_py_test(
    name = "checkpointers_lib_test",
    srcs = ["checkpointers_lib_test.py"],
    deps = [
        ":checkpointers_lib",
        "//reverb:pybind",
    ],
)
