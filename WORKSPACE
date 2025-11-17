workspace(name = "reverb")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TODO: use a released version
http_archive(
    name = "org_tensorflow",
    patch_args = ["-p1"],
    patches = [
        "//third_party:add_core_protos_filegroup.patch",
    ],
    sha256 = "2e1a60686f562914cf3fa6414192be894a163e142058ba30772b3f44188d1cdb",
    strip_prefix = "tensorflow-9da042273660460b94d29062ef02428fc98b0758",
    url = "https://github.com/tensorflow/tensorflow/archive/9da042273660460b94d29062ef02428fc98b0758.zip",
)

# load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
# local_repository(
#     name = "org_tensorflow",
#     path = "../tensorflow"
# )

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "3.11",
    requirements = {
        "3.11": "//reverb/pip_package:requirements_lock_3_11.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "get_toolchain_name_per_python_version", "python_init_toolchains")

python_init_toolchains()

load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

numpy_annotation = package_annotation(
    additive_build_content = """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
name = "numpy_headers_2",
hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
strip_include_prefix="site-packages/numpy/_core/include/",
)
cc_library(
name = "numpy_headers_1",
hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
strip_include_prefix="site-packages/numpy/core/include/",
)
cc_library(
name = "numpy_headers",
deps = [":numpy_headers_2", ":numpy_headers_1"],
)
""",
)

tensorflow_annotation = package_annotation(
    additive_build_content = """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "headers_lib",
    hdrs = glob(
        [
            "site-packages/tensorflow/include/**/*",
        ],
    ),
    strip_include_prefix = "site-packages/tensorflow/include/"
)

cc_library(
    name = "framework_lib",
    srcs = select({
        "@platforms//os:linux": ["site-packages/tensorflow/libtensorflow_framework.so.2"],
        "@platforms//os:macos": ["site-packages/tensorflow/libtensorflow_framework.2.dylib"]
    }),
    visibility = ["//visibility:public"],
)
""",
)

pip_parse(
    name = "pypi",
    annotations = {
        "numpy": numpy_annotation,
        "tensorflow": tensorflow_annotation,
        "tf-nightly": tensorflow_annotation,
    },
    extra_hub_aliases = {
        "numpy": ["numpy_headers"],
        "tf_nightly": [
            "headers_lib",
            "framework_lib",
        ],
        "tensorflow": [
            "headers_lib",
            "framework_lib",
        ],
    },
    python_interpreter_target = "@{}_host//:python".format(
        get_toolchain_name_per_python_version("python"),
    ),
    requirements_lock = REQUIREMENTS,
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "nvidia_wheel_versions_repository",
    "python_wheel_version_suffix_repository",
)

nvidia_wheel_versions_repository(
    name = "nvidia_wheel_versions",
    versions_source = "@org_tensorflow//ci/official/requirements_updater:nvidia-requirements.txt",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64_cuda")

load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)
