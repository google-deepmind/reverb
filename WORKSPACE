workspace(name = "reverb")

load("//tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "832e2f309c57da9c1e6d4542dedd34b24e4192ecb4d62f6f4866a737454c9970",
    strip_prefix = "pybind11-2.10.4",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.4.tar.gz"],
)
http_archive(
    name = "pybind11_bazel",
    sha256 = "6426567481ee345eb48661e7db86adc053881cb4dd39fbf527c8986316b682b9",
    strip_prefix = "pybind11_bazel-fc56ce8a8b51e3dd941139d329b63ccfea1d304b",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/fc56ce8a8b51e3dd941139d329b63ccfea1d304b.zip"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

git_repository(
    name = "com_google_snappy",
    commit = "c9f9edf6d75bb065fa47468bf035e051a57bec7c",
    remote = "https://github.com/google/snappy",
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "5e53505a6c84030a26c4fddd71b3f46feec8e0a8eccff2a903b189d349ca6ff5",
    strip_prefix = "grpc-1.54.0",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.54.0.tar.gz"],
)

ABSL_COMMIT = "273292d1cfc0a94a65082ee350509af1d113344d"

ABSL_SHA256 = "94aef187f688665dc299d09286bfa0d22c4ecb86a80b156dff6aabadc5a5c26d"

http_archive(
    name = "com_google_absl",
    sha256 = ABSL_SHA256,
    strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
    urls = ["https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT)],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()
