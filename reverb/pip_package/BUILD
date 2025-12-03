load("@local_xla//third_party/py:python_wheel.bzl", "collect_data_files", "transitive_py_deps")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//reverb/pip_package:reverb_wheel.bzl", "reverb_wheel")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "licenses",
    data = [
        "//:LICENSE",
    ],
)

compile_pip_requirements(
    name = "requirements",
    srcs = [
        "requirements.in",
    ],
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    generate_hashes = True,
    requirements_txt = REQUIREMENTS,
    tags = ["manual"],
)

exports_files(
    glob(["requirements*.txt"]),
)

collect_data_files(
    name = "cc_deps",
    deps = [
        "//reverb",
    ],
)

transitive_py_deps(
    name = "py_deps",
    deps = [
        "//reverb",
        "//reverb/server_executable:server_from_proto",
        "//reverb/server_executable:server_main",
    ],
)

py_binary(
    name = "build_wheel",
    srcs = ["build_wheel.py"],
    deps = [
        "@pypi//build",
    ],
)

reverb_wheel(
    name = "wheel",
    headers = [],
    platform_name = select({
        "@platforms//os:osx": "macosx",
        "@platforms//os:macos": "macosx",
        "@platforms//os:windows": "win",
        "@platforms//os:linux": "linux",
    }),
    platform_tag = select({
        "@platforms//cpu:aarch64": "arm64",
        "@platforms//cpu:arm64": "arm64",
        "@platforms//cpu:x86_64": "x86_64",
        "@platforms//cpu:ppc": "ppc64le",
    }),
    source_files = [
        "//:README.md",
        # "setup.py",
        ":licenses",
        ":cc_deps",
        ":py_deps",
        ":pyproject.toml",
        ":hatch_build.py",
    ],
    tags = ["manual"],
)
