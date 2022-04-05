workspace(name = "reverb")

# To change to a version of protoc compatible with tensorflow:
#  1. Convert the required header version to a version string, e.g.:
#     3011004 => "3.11.4"
#  2. Calculate the sha256 of the binary:
#     PROTOC_VERSION="3.11.4"
#     curl -L "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" | sha256sum
#  3. Update the two variables below.
#
PROTOC_VERSION = "3.9.0"
PROTOC_SHA256 = "15e395b648a1a6dda8fd66868824a396e9d3e89bc2c8648e3b9ab9801bea5d55"

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
    ],
)

#SIll test of presubmit
http_archive(
    name = "com_google_absl",
    sha256 = "237e2e6aec7571ae90d961d02de19f56861a7417acbbc15713b8926e39d461ed",  # SHARED_ABSL_SHA
    strip_prefix = "abseil-cpp-215105818dfde3174fe799600bb0f3cae233d0bf",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/215105818dfde3174fe799600bb0f3cae233d0bf.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/215105818dfde3174fe799600bb0f3cae233d0bf.tar.gz",
    ],
)

# Note that the Python dependencies are not tracked by bazel here, but
# in setup.py.

## Begin GRPC related deps
http_archive(
    name = "com_github_grpc_grpc",
    patch_cmds = [
        """sed -i.bak 's/"python",/"python3",/g' third_party/py/python_configure.bzl""",
        """sed -i.bak 's/PYTHONHASHSEED=0/PYTHONHASHSEED=0 python3/g' bazel/cython_library.bzl""",
    ],
    sha256 = "39bad059a712c6415b168cb3d922cb0e8c16701b475f047426c81b46577d844b",
    strip_prefix = "grpc-reverb_fix",
    urls = [
        # Patched version of GRPC / boringSSL to make it compile with old TF GCC compiler
        "https://github.com/qstanczyk/grpc/archive/reverb_fix.tar.gz",
    ],
)


load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()


load("@upb//bazel:workspace_deps.bzl", "upb_deps")

upb_deps()

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()
## End GRPC related deps

load(
    "//reverb/cc/platform/default:repo.bzl",
    "cc_tf_configure",
    "reverb_protoc_deps",
    "reverb_python_deps",
)

cc_tf_configure()

reverb_python_deps()

reverb_protoc_deps(version = PROTOC_VERSION, sha256 = PROTOC_SHA256)
