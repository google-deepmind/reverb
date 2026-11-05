load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")
load(":reverb_wheel.bzl", "reverb_wheel")

licenses(["notice"])

py_binary(
    name = "build_wheel",
    srcs = ["build_wheel.py"],
    deps = [
        "@pypi//build",
        "@pypi//hatchling",
    ],
)

py_test(
    name = "build_wheel_test",
    srcs = [
        "build_wheel.py",
        "build_wheel_test.py",
    ],
    imports = ["."],
    main = "build_wheel_test.py",
)

reverb_wheel(
    name = "wheel",
    platform = select({
        "//third_party/bzlmod:is_macos_arm64": "macosx_12_0_arm64",
        "//third_party/bzlmod:is_linux_x86_64": "linux_x86_64",
        "//third_party/bzlmod:is_linux_arm64": "linux_aarch64",
    }),
    source_files = [
        "//:LICENSE",
        "//:README.md",
        "hatch_build.py",
        "pyproject.toml",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "//reverb",
        "//reverb/server_executable:server_from_proto",
        "//reverb/server_executable:server_main",
    ],
)
