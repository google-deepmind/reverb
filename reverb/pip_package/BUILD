licenses(["notice"])  # Apache 2.0

filegroup(
    name = "licenses",
    data = [
        "//:LICENSE",
    ],
)

py_library(
    name = "reverb_version",
    srcs = ["reverb_version.py"],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "setup.py",
        ":licenses",
        ":reverb_version",
        "//reverb",
    ],
)
