licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "setup.py",
        "//reverb",
    ],
)
