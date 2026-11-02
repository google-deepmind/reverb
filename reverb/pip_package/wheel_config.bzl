"""Repository configuration for release and nightly wheel metadata."""

def _wheel_config_impl(ctx):
    name = ctx.getenv("WHEEL_NAME", "dm_reverb")
    wheel_type = ctx.getenv("ML_WHEEL_TYPE", "release")
    date = ctx.getenv("ML_WHEEL_BUILD_DATE", "")
    if wheel_type not in ["release", "nightly"]:
        fail("`ML_WHEEL_TYPE` must be `release` or `nightly`")
    if wheel_type == "nightly" and (len(date) != 8 or not date.isdigit()):
        fail("Nightly wheels require `ML_WHEEL_BUILD_DATE` in `YYYYMMDD` format")
    ctx.file("BUILD.bazel", "exports_files([\"config.bzl\"])\n")
    ctx.file("config.bzl", "WHEEL_NAME = %r\nWHEEL_VERSION_SUFFIX = %r\n" % (
        name,
        ".dev" + date if wheel_type == "nightly" else "",
    ))

wheel_config_repository = repository_rule(implementation = _wheel_config_impl)
