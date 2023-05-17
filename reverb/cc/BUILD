# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library", "py_grpc_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_test(
    name = "chunk_store_test",
    srcs = ["chunk_store_test.cc"],
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "rate_limiter_test",
    srcs = ["rate_limiter_test.cc"],
    deps = [
        ":table",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "table_test",
    srcs = ["table_test.cc"],
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
        ":table",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/support:task_executor",
        "//reverb/cc/table_extensions:interface",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "tensor_compression_test",
    srcs = ["tensor_compression_test.cc"],
    deps = [
        ":tensor_compression",
        "//reverb/cc/testing:tensor_testutil",
    ]
)

cc_test(
    name = "sampler_test",
    srcs = ["sampler_test.cc"],
    deps = [
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":sampler",
        ":table",
        ":tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
        "//reverb/cc/testing:time_testutil",
    ]
)

cc_test(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":client",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":chunker",
        ":client",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
    ],
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    visibility = [
        "//reverb:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@local_config_tf//:tf_header_lib"
    ]
)

cc_library(
    name = "conversions",
    srcs = ["conversions.cc"],
    hdrs = ["conversions.h"],
    deps = [
        "@local_config_tf//:numpy_headers",
        "@local_config_python//:python_headers",
        "@local_config_tf//:tf_header_lib",
    ]
)

cc_library(
    name = "chunk_store",
    srcs = ["chunk_store.cc"],
    hdrs = ["chunk_store.h"],
    deps = [
        ":schema_cc_proto",
        "//reverb/cc/platform:hash_map",
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/status",
    ]
)

cc_library(
    name = "table",
    srcs = [
        "rate_limiter.cc",
        "table.cc",
    ],
    hdrs = [
        "rate_limiter.h",
        "table.h",
    ],
    visibility = ["//reverb:__subpackages__"],
    deps = depset([
        ":chunk_store",
        ":errors",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/support:state_statistics",
        "//reverb/cc/support:task_executor",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/table_extensions:interface",
    ]),
)

cc_library(
    name = "tensor_compression",
    srcs = ["tensor_compression.cc"],
    hdrs = ["tensor_compression.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:snappy",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/flags:flag",
        "@local_config_tf//:tf_header_lib",
    ]
)

cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    deps = depset([
        ":chunk_store",
        ":errors",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
    ]),
)

cc_library(
    name = "writer",
    srcs = ["writer.cc"],
    hdrs = ["writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = depset([
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/functional:bind_front",
    ]),
)

cc_library(
    name = "trajectory_writer",
    srcs = ["trajectory_writer.cc"],
    hdrs = ["trajectory_writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = depset([
        ":chunker",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/random",
    ]),
)

cc_library(
    name = "streaming_trajectory_writer",
    srcs = ["streaming_trajectory_writer.cc"],
    hdrs = ["streaming_trajectory_writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = depset([
        ":chunker",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
    ]),
)

cc_test(
    name = "trajectory_writer_test",
    srcs = ["trajectory_writer_test.cc"],
    deps = [
        ":chunker",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ]
)

cc_test(
    name = "streaming_trajectory_writer_test",
    srcs = ["streaming_trajectory_writer_test.cc"],
    deps = [
        ":chunker",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":streaming_trajectory_writer",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:queue",
        "//reverb/cc/support:signature",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ]
)

cc_library(
    name = "chunker",
    srcs = ["chunker.cc"],
    hdrs = ["chunker.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = depset([
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
        "@com_google_absl//absl/random",
    ]),
)

cc_test(
    name = "chunker_test",
    srcs = ["chunker_test.cc"],
    deps = [
        ":chunker",
        ":schema_cc_proto",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:signature",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ]
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":chunker",
        ":patterns_cc_proto",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":sampler",
        ":schema_cc_proto",
        ":streaming_trajectory_writer",
        ":structured_writer",
        ":trajectory_writer",
        ":writer",
        "//reverb/cc/platform:grpc_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:uint128",
        "@com_google_absl//absl/random",
    ]
)

proto_library(
    name = "schema_proto",
    srcs = ["schema.proto"],
    deps = [
        "@local_config_tf//:tensor_proto",
        "@local_config_tf//:struct_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
    # visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "schema_cc_proto",
    deps = [":schema_proto"],
)

py_proto_library(
    name = "schema_py_pb2",
    deps = [":schema_proto"],
)

proto_library(
    name = "patterns_proto",
    srcs = ["patterns.proto"],
    deps = [
        "@local_config_tf//:tensor_proto",
        "@local_config_tf//:struct_proto",
        "@local_config_tf//:resource_handle_proto",
        "@local_config_tf//:tensor_shape_proto",
        "@local_config_tf//:types_proto",
    ]
)

cc_proto_library(
    name = "patterns_cc_proto",
    deps = [":patterns_proto"],
)

py_proto_library(
    name = "patterns_py_pb2",
    deps = [":patterns_proto"],
)

proto_library(
    name = "reverb_service_proto",
    srcs = ["reverb_service.proto"],
    deps = [
        ":schema_proto",
        "@local_config_tf//:tensor_proto",
        "@local_config_tf//:struct_proto",
        "@local_config_tf//:resource_handle_proto",
        "@local_config_tf//:tensor_shape_proto",
        "@local_config_tf//:types_proto",
    ],
)

cc_proto_library(
    name = "reverb_service_cc_proto",
    deps = [":reverb_service_proto"],
)

cc_grpc_library(
    name = "reverb_service_cc_grpc",
    srcs = [":reverb_service_proto"],
    grpc_only = True,
    generate_mocks = True,
    visibility = [
        "//reverb:__subpackages__",
    ],
    deps = [":reverb_service_cc_proto"],
)

py_proto_library(
    name = "reverb_service_py_pb2",
    deps = [":reverb_service_proto"],
)

py_grpc_library(
    name = "reverb_service_py_pb2_grpc",
    srcs = [":reverb_service_proto"],
    deps = [":reverb_service_py_pb2"],
)


cc_library(
    name = "reverb_service_impl",
    srcs = ["reverb_service_impl.cc"],
    hdrs = [
        "reverb_service_impl.h",
    ],
    deps = [
        ":reverb_server_reactor",
        ":reverb_service_cc_grpc",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":table",
        ":task_worker",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/flags:flag",
    ],
    alwayslink = True,
)

cc_library(
    name = "task_worker",
    hdrs = ["task_worker.h"],
    deps = [
        ":thread_stats",
        "//reverb/cc:table",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:periodic_closure",
        "//reverb/cc/support:unbounded_queue",
    ]
)

cc_library(
    name = "reverb_server_reactor",
    hdrs = ["reverb_server_reactor.h"],
    deps = [
        ":task_worker",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
    ]
)

cc_library(
    name = "thread_stats",
    srcs = ["thread_stats.cc"],
    hdrs = ["thread_stats.h"],
    deps = [
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "reverb_service_impl_test",
    srcs = ["reverb_service_impl_test.cc"],
    deps = [
        ":reverb_service_cc_proto",
        ":reverb_service_impl",
        ":schema_cc_proto",
        ":task_worker",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/testing:proto_test_util",
    ]
)

cc_test(
    name = "thread_stats_test",
    srcs = ["thread_stats_test.cc"],
    deps = [
        ":thread_stats",
    ]
)

cc_test(
    name = "structured_writer_test",
    srcs = ["structured_writer_test.cc"],
    deps = [
        ":chunker",
        ":patterns_cc_proto",
        ":structured_writer",
        ":trajectory_writer",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:signature",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ]
)

cc_library(
    name = "structured_writer",
    srcs = ["structured_writer.cc"],
    hdrs = ["structured_writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = depset([
        ":chunker",
        ":patterns_cc_proto",
        ":trajectory_writer",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "@com_google_absl//absl/status:statusor",
    ]),
)
