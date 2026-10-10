"""Source generation with the Reverb module's native Protobuf toolchain."""

def _generate_impl(ctx):
    source = ctx.file.src
    workspace = source.owner.workspace_root or "."
    tensorflow = ctx.files.tensorflow_protos[0]
    tensorflow_root = "tensorflow/core=" + tensorflow.owner.workspace_root
    protobuf = ctx.files.well_known[0]
    protobuf_root = protobuf.path.split("/google/protobuf/")[0]
    arguments = ctx.actions.args()
    arguments.add_all(["-I" + workspace, "-I" + tensorflow_root, "-I" + protobuf_root])
    output_root = ctx.bin_dir.path + "/" + workspace
    arguments.add("--" + ctx.attr.language + "_out=" + output_root)
    if ctx.attr.language == "grpc":
        arguments.add("--plugin=protoc-gen-grpc=" + ctx.executable.plugin.path)
        if ctx.attr.generate_mocks:
            arguments.add("--grpc_opt=generate_mock_code=true")
    arguments.add(source.path)
    ctx.actions.run(
        executable = ctx.executable.protoc,
        arguments = [arguments],
        inputs = depset(ctx.files.src + ctx.files.proto_deps + ctx.files.tensorflow_protos + ctx.files.well_known),
        tools = [ctx.executable.plugin] if ctx.attr.language == "grpc" else [],
        outputs = ctx.outputs.outs,
        mnemonic = "ReverbProto",
    )
    return [DefaultInfo(files = depset(ctx.outputs.outs))]

reverb_generate_proto = rule(
    implementation = _generate_impl,
    attrs = {
        "src": attr.label(allow_single_file = [".proto"], mandatory = True),
        "proto_deps": attr.label_list(allow_files = True),
        "tensorflow_protos": attr.label(allow_files = True, mandatory = True),
        "well_known": attr.label(allow_files = True, mandatory = True),
        "protoc": attr.label(executable = True, cfg = "exec", mandatory = True),
        "plugin": attr.label(executable = True, cfg = "exec"),
        "language": attr.string(values = ["cpp", "python", "grpc"]),
        "generate_mocks": attr.bool(),
        "outs": attr.output_list(mandatory = True),
    },
)
