"""TensorFlow schema inputs for source-built Reverb."""

def _tensorflow_protos_impl(ctx):
    ctx.download_and_extract(
        url = "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.21.0.tar.gz",
        sha256 = "ef3568bb4865d6c1b2564fb5689c19b6b9a5311572cd1f2ff9198636a8520921",
        stripPrefix = "tensorflow-2.21.0/tensorflow/core",
    )
    ctx.delete("framework/BUILD")
    ctx.delete("protobuf/BUILD")
    ctx.download(
        url = "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.21.0/LICENSE",
        sha256 = "71c6915d04265772a0339bed47276942c678b45cc01534210ebe6984fd1aec65",
        output = "LICENSE",
    )
    ctx.file("BUILD.bazel", '''filegroup(
    name = "protos",
    srcs = [
        "framework/resource_handle.proto",
        "framework/tensor.proto",
        "framework/tensor_shape.proto",
        "framework/types.proto",
        "protobuf/struct.proto",
    ],
    visibility = ["//visibility:public"],
)
''')

tensorflow_protos_repository = repository_rule(implementation = _tensorflow_protos_impl)

def _pybind11_impl(ctx):
    ctx.file("BUILD.bazel", 'alias(name = "pybind11", actual = "{}", visibility = ["//visibility:public"])'.format(ctx.attr.actual))

pybind11_repository = repository_rule(
    implementation = _pybind11_impl,
    attrs = {"actual": attr.label(mandatory = True)},
)
