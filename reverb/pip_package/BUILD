licenses(["notice"])  # Apache 2.0

filegroup(
    name = "licenses",
    data = [
        "//:LICENSE",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "licenses",
        "setup.py",
        "//reverb",
    ],
)
