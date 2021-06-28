# Description: Reverb is an efficient and easy to use prioritized replay system designed for ML research.

load(
    "//reverb/cc/platform:build_rules.bzl",
    "reverb_absl_deps",
    "reverb_cc_grpc_library",
    "reverb_cc_library",
    "reverb_cc_proto_library",
    "reverb_cc_test",
    "reverb_grpc_deps",
    "reverb_py_proto_library",
    "reverb_tf_deps",
)

package(default_visibility = ["//reverb:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

reverb_cc_test(
    name = "chunk_store_test",
    srcs = ["chunk_store_test.cc"],
    deps = [
        ":chunk_store",
        ":schema_cc_proto",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps(),
)

reverb_cc_test(
    name = "rate_limiter_test",
    srcs = ["rate_limiter_test.cc"],
    deps = [
        ":table",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_absl_deps(),
)

reverb_cc_test(
    name = "table_test",
    srcs = ["table_test.cc"],
    deps = [
        ":chunk_store",
        ":table",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/table_extensions:interface",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "tensor_compression_test",
    srcs = ["tensor_compression_test.cc"],
    deps = [
        ":tensor_compression",
        "//reverb/cc/testing:tensor_testutil",
    ] + reverb_tf_deps(),
)

reverb_cc_test(
    name = "sampler_test",
    srcs = ["sampler_test.cc"],
    deps = [
        ":sampler",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":table",
        ":tensor_compression",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
        "//reverb/cc/testing:time_testutil",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":client",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":writer",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "//reverb/cc/support:queue",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":client",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":trajectory_writer",
        ":chunker",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/support:uint128",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_grpc_deps(),
)

reverb_cc_test(
    name = "reverb_service_impl_test",
    srcs = ["reverb_service_impl_test.cc"],
    deps = [
        ":reverb_service_cc_proto",
        ":reverb_service_impl",
        ":schema_cc_proto",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    visibility = [
        "//reverb:__subpackages__",
    ],
    deps = reverb_absl_deps(),
)

reverb_cc_library(
    name = "chunk_store",
    srcs = ["chunk_store.cc"],
    hdrs = ["chunk_store.h"],
    deps = [
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:queue",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
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
    deps = [
        ":chunk_store",
        ":errors",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:checkpoint_cc_proto",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/table_extensions:interface",
    ] + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_library(
    name = "tensor_compression",
    srcs = ["tensor_compression.cc"],
    hdrs = ["tensor_compression.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:snappy",
    ] + reverb_tf_deps(),
)

reverb_cc_library(
    name = "sampler",
    srcs = ["sampler.cc"],
    hdrs = ["sampler.h"],
    deps = [
        ":chunk_store",
        ":errors",
        ":reverb_service_cc_grpc_proto",
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
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:queue",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "writer",
    srcs = ["writer.cc"],
    hdrs = ["writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "trajectory_writer",
    srcs = ["trajectory_writer.cc"],
    hdrs = ["trajectory_writer.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":chunker",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:trajectory_util",
    ] + reverb_tf_deps() + reverb_absl_deps() + reverb_grpc_deps(),
)

reverb_cc_test(
    name = "trajectory_writer_test",
    srcs = ["trajectory_writer_test.cc"],
    deps = [
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":trajectory_writer",
        ":chunker",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:queue",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/testing:proto_test_util",
        "//reverb/cc/testing:tensor_testutil",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "chunker",
    srcs = ["chunker.cc"],
    hdrs = ["chunker.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":schema_cc_proto",
        ":tensor_compression",
        "//reverb/cc/support:signature",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:tf_util",
        "//reverb/cc/support:key_generators",
        "//reverb/cc/support:trajectory_util",
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_test(
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
    ] + reverb_tf_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    visibility = ["//reverb:__subpackages__"],
    deps = [
        ":sampler",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":schema_cc_proto",
        ":trajectory_writer",
        ":writer",
        "//reverb/cc/platform:grpc_utils",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:signature",
        "//reverb/cc/support:uint128",
    ] + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "reverb_service_impl",
    srcs = ["reverb_service_impl.cc"],
    hdrs = ["reverb_service_impl.h"],
    deps = [
        ":chunk_store",
        ":table",
        ":reverb_service_cc_grpc_proto",
        ":reverb_service_cc_proto",
        ":sampler",
        ":schema_cc_proto",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:thread",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "//reverb/cc/support:queue",
    ] + reverb_tf_deps() + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_proto_library(
    name = "schema_cc_proto",
    srcs = ["schema.proto"],
)

reverb_py_proto_library(
    name = "schema_py_pb2",
    srcs = ["schema.proto"],
    deps = [":schema_cc_proto"],
)

reverb_cc_proto_library(
    name = "reverb_service_cc_proto",
    srcs = ["reverb_service.proto"],
    visibility = ["//reverb:__subpackages__"],
    deps = [":schema_cc_proto"],
)

reverb_cc_grpc_library(
    name = "reverb_service_cc_grpc_proto",
    srcs = ["reverb_service.proto"],
    generate_mocks = True,
    visibility = ["//reverb:__subpackages__"],
    deps = [":reverb_service_cc_proto"],
)

reverb_cc_library(
    name = "reverb_service_async_impl",
    srcs = ["reverb_service_async_impl.cc"],
    hdrs = [
        "reverb_service_async_impl.h",
    ],
    deps = [
        ":reverb_server_bidi_reactor",
        ":reverb_server_table_reactor",
        ":task_worker",
        "//reverb/cc:chunk_store",
        "//reverb/cc:reverb_service_cc_grpc_proto",
        "//reverb/cc:reverb_service_cc_proto",
        "//reverb/cc:sampler",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc:table",
        "//reverb/cc/checkpointing:interface",
        "//reverb/cc/platform:hash_map",
        "//reverb/cc/platform:hash_set",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:thread",
        "//reverb/cc/support:cleanup",
        "//reverb/cc/support:grpc_util",
        "//reverb/cc/support:periodic_closure",
        "//reverb/cc/support:trajectory_util",
        "//reverb/cc/support:uint128",
        "//reverb/cc/support:unbounded_queue",
    ] + reverb_grpc_deps() + reverb_absl_deps(),
    alwayslink = 1,
)

reverb_cc_library(
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
    ] + reverb_absl_deps(),
)

reverb_cc_library(
    name = "reverb_server_bidi_reactor",
    hdrs = ["reverb_server_bidi_reactor.h"],
    deps = [
        ":task_worker",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
    ] + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "reverb_server_table_reactor",
    hdrs = ["reverb_server_table_reactor.h"],
    deps = [
        ":task_worker",
        "//reverb/cc/platform:logging",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/support:grpc_util",
    ] + reverb_grpc_deps() + reverb_absl_deps(),
)

reverb_cc_library(
    name = "thread_stats",
    srcs = ["thread_stats.cc"],
    hdrs = ["thread_stats.h"],
    deps = reverb_absl_deps(),
)

reverb_cc_test(
    name = "reverb_service_async_impl_test",
    srcs = ["reverb_service_async_impl_test.cc"],
    deps = [
        ":reverb_service_async_impl",
        "//reverb/cc:reverb_service_cc_proto",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_grpc_deps() + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_test(
    name = "reverb_service_async_impl_test_table_reactor",
    srcs = ["reverb_service_async_impl_test.cc"],
    args = [
        "--reverb_insert_worker_num_threads=0",
        "--disable_forwards_table_error_test",
    ],
    deps = [
        ":reverb_service_async_impl",
        "//reverb/cc:reverb_service_cc_proto",
        "//reverb/cc:schema_cc_proto",
        "//reverb/cc/platform:checkpointing",
        "//reverb/cc/platform:status_macros",
        "//reverb/cc/platform:status_matchers",
        "//reverb/cc/platform:thread",
        "//reverb/cc/selectors:fifo",
        "//reverb/cc/selectors:interface",
        "//reverb/cc/selectors:uniform",
        "//reverb/cc/testing:proto_test_util",
    ] + reverb_grpc_deps() + reverb_absl_deps() + reverb_tf_deps(),
)

reverb_cc_test(
    name = "thread_stats_test",
    srcs = ["thread_stats_test.cc"],
    deps = [
        ":thread_stats",
    ] + reverb_absl_deps(),
)
