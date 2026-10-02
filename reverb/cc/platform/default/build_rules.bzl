"""Default versions of reverb build rule helpers."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//third_party/bzlmod:proto.bzl", "reverb_generate_proto")

def tf_copts():
    return ["-Wno-sign-compare", "-std=c++17", "-DNDEBUG", "-DEIGEN_MAX_ALIGN_BYTES=64"] + select({
        "//third_party/bzlmod:is_linux_x86_64": ["-mavx"],
        "//third_party/bzlmod:is_macos_arm64": ["-mmacosx-version-min=12.0"],
        "//conditions:default": [],
    })

def reverb_cc_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        testonly = 0,
        **kwargs):
    if testonly:
        new_deps = [
            "@com_google_googletest//:gtest",
        ] + reverb_tf_deps()
    else:
        new_deps = []
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = tf_copts(),
        testonly = testonly,
        deps = depset(deps + new_deps).to_list(),
        **kwargs
    )

def reverb_kernel_library(name, srcs = [], deps = [], **kwargs):
    deps = deps + reverb_tf_deps()
    reverb_cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        alwayslink = 1,
        **kwargs
    )

reverb_cc_shared_library = cc_shared_library

def _removesuffix(x, txt):
    """Backport of x._removesuffix(txt) for Python version earlier than 3.9."""
    if x.endswith(txt):
        return x[:-len(txt)]
    return x

def _normalize_proto(x):
    return x.removesuffix("_proto").removesuffix("_cc") + "_proto"

def _filegroup_name(x):
    return _normalize_proto(x) + "_filegroup"

def _generate(name, srcs, deps, language, generate_mocks = False, **kwargs):
    proto_deps = [_filegroup_name(x) for x in deps if x.endswith("_proto")]
    native.filegroup(name = _filegroup_name(name), srcs = srcs + proto_deps, **kwargs)
    outputs = []
    for i, source in enumerate(srcs):
        stem = source.removesuffix(".proto")
        if language == "cpp":
            generated = [stem + ".pb.cc", stem + ".pb.h"]
        elif language == "python":
            generated = [stem + "_pb2.py"]
        else:
            generated = [stem + ".grpc.pb.cc", stem + ".grpc.pb.h"]
            if generate_mocks:
                generated.append(stem + "_mock.grpc.pb.h")
        reverb_generate_proto(
            name = name + "_generate_" + str(i),
            src = source,
            proto_deps = proto_deps,
            tensorflow_protos = "//third_party/bzlmod:tensorflow_protos",
            well_known = "@com_google_protobuf//:well_known_type_protos",
            protoc = "@com_google_protobuf//:protoc",
            plugin = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin" if language == "grpc" else None,
            language = language,
            generate_mocks = generate_mocks,
            outs = generated,
            testonly = kwargs.get("testonly", False),
        )
        outputs.extend(generated)
    return outputs

def reverb_cc_proto_library(name, srcs = [], deps = [], **kwargs):
    outputs = _generate(name, srcs, deps, "cpp", **kwargs)
    reverb_cc_library(
        name = name,
        srcs = [x for x in outputs if x.endswith(".cc")],
        hdrs = [x for x in outputs if x.endswith(".h")],
        deps = deps + reverb_tf_deps(),
        alwayslink = True,
        **kwargs
    )

def reverb_py_proto_library(name, srcs = [], deps = [], **kwargs):
    outputs = _generate(name, srcs, deps, "python", **kwargs)
    py_library(
        name = name,
        srcs = outputs,
        deps = [x for x in deps if not x.endswith("_proto")] + reverb_py_standard_imports(),
        **kwargs
    )

def reverb_cc_grpc_library(name, srcs = [], deps = [], generate_mocks = False, **kwargs):
    outputs = _generate(name, srcs, deps, "grpc", generate_mocks, **kwargs)
    reverb_cc_library(
        name = name,
        srcs = [x for x in outputs if x.endswith(".cc")],
        hdrs = [x for x in outputs if x.endswith(".h")],
        deps = deps + ["@com_github_grpc_grpc//:grpc++_codegen_proto"],
        **kwargs
    )

def reverb_cc_test(name, srcs, deps = [], **kwargs):
    """Reverb-specific version of cc_test.

    Args:
      name: Target name.
      srcs: Target sources.
      deps: Target deps.
      **kwargs: Additional args to cc_test.
    """
    new_deps = [
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:status_matchers",
    ] + reverb_tf_deps()
    size = kwargs.pop("size", "small")
    cc_test(
        name = name,
        size = size,
        copts = tf_copts(),
        srcs = srcs,
        deps = depset(deps + new_deps).to_list(),
        **kwargs
    )

def reverb_gen_op_wrapper_py(name, out, kernel_lib, ops_lib = None, linkopts = [], **kwargs):
    """Generates the py_library `name` with a data dep on the ops in kernel_lib.

    The resulting py_library creates file `$out`, and has a dependency on a
    symbolic library called lib{$name}_gen_op.so, which contains the kernels
    and ops and can be loaded via `tf.load_op_library`.

    Args:
      name: The name of the py_library.
      out: The name of the python file.  Use "gen_{name}_ops.py".
      kernel_lib: A cc_kernel_library kernel target to generate for.
      ops_lib: A cc_kernel_library ops target to generate for.
      linkopts: Forwarded to the `cc_binary` internal target.
      **kwargs: Any args to the `cc_binary` and `py_library` internal rules.
    """
    if not out.endswith(".py"):
        fail("Argument out must end with '.py', but saw: {}".format(out))

    module_name = "lib{}_gen_op".format(name)
    exported_symbols_file = "%s-exported-symbols.lds" % module_name

    # gen_client_ops -> reverb_client
    symbol = "reverb_{}".format(name.split("_")[1])
    native.genrule(
        name = module_name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '*%s*' >$@" % symbol,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )
    version_script_file = "%s-version-script.lds" % module_name
    native.genrule(
        name = module_name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n *%s*;\n local: *;};' >$@" % symbol,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )
    cc_binary(
        name = "{}.so".format(module_name),
        deps = [kernel_lib] + ([ops_lib] if ops_lib else []),
        copts = tf_copts() + [
            "-fno-strict-aliasing",  # allow a wider range of code [aliasing] to compile.
            "-fvisibility=hidden",  # avoid symbol clashes between DSOs.
        ],
        additional_linker_inputs = [
            exported_symbols_file,
            version_script_file,
        ],
        dynamic_deps = ["//reverb:libreverb"],
        linkshared = 1,
        linkopts = linkopts + _rpath_linkopts(module_name) + select({
            "@platforms//os:macos": [
                "-Wl,-undefined,dynamic_lookup",
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//conditions:default": [
                "-Wl,--version-script,$(location %s)" % version_script_file,
            ],
        }),
        **kwargs
    )
    native.genrule(
        name = "{}_genrule".format(out),
        outs = [out],
        cmd = """echo 'import tensorflow as _tf
from reverb.platform.default import load_op_library as _load_op_library

try:
  _reverb_gen_op = _tf.load_op_library(
    _tf.compat.v1.resource_loader.get_path_to_datafile("lib{}_gen_op.so"))
except _tf.errors.NotFoundError as e:
  _load_op_library.reraise_wrapped_error(e)
_locals = locals()
for k in dir(_reverb_gen_op):
  _locals[k] = getattr(_reverb_gen_op, k)
del _locals' > $@""".format(name),
    )
    deps = kwargs.pop("deps", [])
    deps.append("//reverb/platform/default:load_op_library")
    py_library(
        name = name,
        srcs = [out],
        data = [":lib{}_gen_op.so".format(name)],
        deps = deps,
        **kwargs
    )

def reverb_py_proto_deps():
    return []

def reverb_pytype_library(deps = [], **kwargs):
    if "strict_deps" in kwargs:
        kwargs.pop("strict_deps")
    py_library(
        deps = deps + reverb_py_standard_imports() + reverb_py_proto_deps(),
        **kwargs
    )

reverb_pytype_strict_library = reverb_pytype_library

def reverb_pytype_binary(deps = [], **kwargs):
    if "strict_deps" in kwargs:
        kwargs.pop("strict_deps")
    py_binary(
        deps = deps + reverb_py_standard_imports() + reverb_py_proto_deps(),
        **kwargs
    )

reverb_pytype_strict_binary = reverb_pytype_binary

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        "@platforms//os:macos": [
            "-Wl,-rpath,@loader_path/%s/tensorflow" % "/".join([".."] * (levels_to_root + 1)),
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
        ],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
            "-Wl,-rpath,$$ORIGIN/%s/tensorflow" % "/".join([".."] * (levels_to_root + 1)),
        ],
    })

def reverb_pybind_extension(
        name,
        srcs,
        module_name,
        hdrs = [],
        features = [],
        srcs_version = "PY3",
        data = [],
        copts = [],
        linkopts = [],
        deps = [],
        defines = [],
        visibility = None,
        testonly = None,
        licenses = None,
        compatible_with = None,
        restricted_to = None,
        deprecation = None,
        pytype_srcs = None):
    """Builds a generic Python extension module.

    The module can be loaded in python by performing "import ${name}.".

    Args:
      name: Name.
      srcs: cc files.
      module_name: The name of the hidden module.  It should be different
        from `name`, and *must* match the MODULE declaration in the .cc file.
      hdrs: h files.
      features: see bazel docs.
      srcs_version: srcs_version for py_library.
      data: data deps.
      copts: compilation opts.
      linkopts: linking opts.
      deps: cc_library deps.
      defines: cc_library defines.
      visibility: visibility.
      testonly: whether the rule is testonly.
      licenses: see bazel docs.
      compatible_with: see bazel docs.
      restricted_to: see bazel docs.
      deprecation:  see bazel docs.
      pytype_srcs: Unused list of pytype stub files.
    """
    if name == module_name:
        fail(
            "Must have name != module_name ({} vs. {}) because the python ".format(name, module_name) +
            "wrapper $name.py needs to add extra logic loading tensorflow.",
        )
    py_file = "%s.py" % name
    so_file = "%s.so" % module_name
    pyd_file = "%s.pyd" % module_name
    symbol = "init%s" % module_name
    symbol2 = "init_%s" % module_name
    symbol3 = "PyInit_%s" % module_name
    exported_symbols_file = "%s-exported-symbols.lds" % module_name
    version_script_file = "%s-version-script.lds" % module_name
    native.genrule(
        name = module_name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '_%s\n' >$@" % (symbol3),
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )
    native.genrule(
        name = module_name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n %s;\n local: *;};' >$@" % (symbol3),
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
        testonly = testonly,
    )
    cc_binary(
        name = so_file,
        srcs = srcs + hdrs,
        data = data,
        copts = copts + tf_copts() + [
            "-fno-strict-aliasing",  # allow a wider range of code [aliasing] to compile.
            "-fexceptions",  # pybind relies on exceptions, required to compile.
            "-fvisibility=hidden",  # avoid pybind symbol clashes between DSOs.
        ],
        linkopts = linkopts + _rpath_linkopts(module_name) + select({
            "@platforms//os:macos": [
                "-Wl,-undefined,dynamic_lookup",
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//conditions:default": [
                "-Wl,--version-script,$(location %s)" % version_script_file,
            ],
        }),
        deps = deps,
        additional_linker_inputs = [
            exported_symbols_file,
            version_script_file,
        ],
        dynamic_deps = ["//reverb:libreverb"],
        defines = defines,
        features = features + ["-use_header_modules"],
        linkshared = 1,
        testonly = testonly,
        licenses = licenses,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.genrule(
        name = module_name + "_pyd_copy",
        srcs = [so_file],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )
    native.genrule(
        name = name + "_py_file",
        outs = [py_file],
        cmd = """echo 'import tensorflow as _tf
from reverb.platform.default import load_op_library as _load_op_library
try:
  from .%s import *
except ImportError as e:
  _load_op_library.reraise_wrapped_error(e)
del _tf' >$@""" % module_name,
        output_licenses = ["unencumbered"],
        visibility = visibility,
        testonly = testonly,
    )
    py_library(
        name = name,
        data = [so_file, "//reverb:libreverb"],
        deps = ["//reverb/platform/default:load_op_library"],
        srcs = [py_file],
        srcs_version = srcs_version,
        licenses = licenses,
        testonly = testonly,
        visibility = visibility,
        deprecation = deprecation,
        restricted_to = restricted_to,
        compatible_with = compatible_with,
    )

def reverb_py_standard_imports():
    return ["//third_party/bzlmod:python_dependencies"]

def reverb_py_test(
        name,
        srcs = [],
        deps = [],
        paropts = [],
        python_version = "PY3",
        **kwargs):
    size = kwargs.pop("size", "small")
    if "enable_dashboard" in kwargs:
        kwargs.pop("enable_dashboard")
    py_test(
        name = name,
        size = size,
        srcs = srcs,
        deps = deps + reverb_py_standard_imports() + reverb_py_proto_deps(),
        python_version = python_version,
        **kwargs
    )
    return

def reverb_pybind_deps():
    return ["//third_party/bzlmod:binding_headers"]

def reverb_tf_ops_visibility():
    return [
        "//reverb:__subpackages__",
    ]

def reverb_tf_deps():
    return [
        "//third_party/bzlmod:tensorflow",
    ]

def reverb_grpc_deps():
    return ["@com_github_grpc_grpc//:grpc++"]
