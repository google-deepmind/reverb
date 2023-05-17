package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = [":libtensorflow_framework.so"],
    #data = ["lib/libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "tensor_proto",
    srcs = ["include/tensorflow/core/framework/tensor.proto"],
    strip_import_prefix = "include/",
    deps = [
        ":resource_handle_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "struct_proto",
    srcs = ["include/tensorflow/core/protobuf/struct.proto"],
    strip_import_prefix = "include/",
    deps = [
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "resource_handle_proto",
    srcs = ["include/tensorflow/core/framework/resource_handle.proto"],
    strip_import_prefix = "include/",
    deps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "tensor_shape_proto",
    srcs = ["include/tensorflow/core/framework/tensor_shape.proto"],
    strip_import_prefix = "include/",
    visibility = ["//visibility:public"],
)

proto_library(
    name = "types_proto",
    srcs = ["include/tensorflow/core/framework/types.proto"],
    strip_import_prefix = "include/",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "numpy_headers",
    hdrs = [":numpy_include"],
    includes = ["numpy_include"],
    strip_include_prefix = "numpy_include/",
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
%{NUMPY_INCLUDE_GENRULE}
