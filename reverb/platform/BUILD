load("@rules_python//python:defs.bzl", "py_library", "py_test")
package(default_visibility = [":__subpackages__"])

licenses(["notice"])

py_library(
    name = "checkpointers_lib",
    srcs = ["checkpointers_lib.py"],
    srcs_version = "PY3",
    data = ["//reverb:pybind.so"],
)

py_test(
    name = "checkpointers_lib_test",
    srcs = ["checkpointers_lib_test.py"],
    python_version = "PY3",
    deps = [":checkpointers_lib"],
    data = ["//reverb:pybind.so"],
)
