# How to develop and build Reverb with the Docker containers

## Overview

This document covers a couple scenarios:

 *  <a href='#Release'>Create a Reverb release</a>
 *  <a href='#Develop'>Develop Reverb inside a Docker container</a>
 *  <a href='#builds-tips-and-hints'>Build tips and hints</a>

While there is overlap in the above scenarios, treating them separately seems
the most clear at the moment. Before you get started, setup a local variable
pointing to your local Reverb GitHub repo.

```shell
$ export REVERB_DIR=/path/to/reverb/github/repo
```

<a id='Release'></a>
## Create a stable Reverb release

There are two steps for building the Reverb package.

  * Build the Docker container with the version of TensorFlow to build Reverb
    against.
  * Execute the build and declare any specific TensorFlow dependency for the
    pip install. The dependency is only enforced if the user uses
    `pip install reverb[tensorflow]`.

Execute from the root of the git repository. The end result will end up in
`$REVERB_DIR/dist`.

```shell

##################################
# Creates the Docker container.
##################################
# Builds the container with Python 3.8, 3.9, 3.10, and 3.11. Set the
# `build-arg tensorflow_pip` to the version of TensorFlow to build against.
# If building against an RC, use == rather than ~= for `tensorflow_pip`.
$ docker build --pull --no-cache \
  --tag tensorflow:reverb_release \
  --build-arg tensorflow_pip=tensorflow~=2.8.0 \
  --build-arg python_version="python3.8 python3.9 python3.10 python3.11" \
  - < "$REVERB_DIR/docker/release.dockerfile"

#################################################
# Builds Reverb against TensorFlow stable or rc.
#################################################

# Builds Reverb against most recent stable release of TensorFlow and
# requires `tensorflow~=2.8.0` if using `pip install reverb[tensorflow]`.
# Packages for Python 3.8, 3.9, 3.10, and 3.11 are created.
$ docker run --rm --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
  tensorflow:reverb_release bash oss_build.sh --clean true \
  --tf_dep_override "tensorflow~=2.11.0" --release --python "3.8 3.9 3.10 3.11"

# Builds Reverb against an RC of TensorFlow. `>=` and `~=` are not effective
# because pip does not recognize 2.4.0rc0 as greater than 2.3.0. RC builds need
# to have a strict dependency on the RC of TensorFlow used.
$ docker run --rm --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
  tensorflow:reverb_release bash oss_build.sh --clean true \
  --tf_dep_override "tensorflow==2.12.0rc0" --release --python "3.8 3.9 3.10 3.11"

# Builds a debug version of Reverb. The debug version is not labeled as debug
# as that can result in a user installing both the debug and regular packages
# making it unclear which is installed as they both have the same package
# namespace. The command below puts the .whl files in ./dist/debug/**.
# Debug builds are ~90M compared to normal builds that are closer to 7M.
$ docker run --rm --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
  tensorflow:reverb_release bash oss_build.sh --clean true --debug_build true \
  --output_dir /tmp/reverb/dist/debug/ --tf_dep_override "tensorflow~=2.10.0" \
  --release --python "3.8 3.9 3.10 3.11"

```

<a id='Develop'></a>
## Develop Reverb inside a Docker container

1. Build the Docker container. By default the container is setup for python 3.9.
   Use the `python_version` arg to configure the container with 3.8 or 3.9.

  ```shell
  $ docker build --tag tensorflow:reverb - < "$REVERB_DIR/docker/release.dockerfile"

  # Alternatively you can build the container with Python 3.8 support.
  $ docker build --tag tensorflow:reverb \
      --build-arg python_version=python3.8 \
      - < "$REVERB_DIR/docker/release.dockerfile"
  ```

1. Run and enter the Docker container.

  ```shell
  $ docker run --rm -it \
    --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
    --mount "type=bind,src=$HOME/.gitconfig,dst=/etc/gitconfig,ro" \
    --name reverb tensorflow:reverb bash
  ```

1. Compile Reverb.

  ```shell
  $ python3.8 configure.py
  $ bazel build -c opt //reverb/pip_package:build_pip_package
  ```

1. Build the .whl file and output it to `/tmp/reverb_build/dist/`.

  ```shell
  $ ./bazel-bin/reverb/pip_package/build_pip_package \
    --dst /tmp/reverb_build/dist/
  ```

1. Install the .whl file.

  ```shell
  # If multiple versions were built, pass the exact wheel to install rather than
  # *.whl.
  $ $PYTHON_BIN_PATH -mpip install --upgrade /tmp/reverb_build/dist/*.whl
  ```

<a id='#builds-tips-and-hints'></a>

## Builds Tips and Hints

### protoc / protobuf version mismatch

There is a
[check in the Reverb build process](https://github.com/deepmind/reverb/blob/master/third_party/protobuf.BUILD)
that checks if the protoc library in Tensorflow matches what Reverb is using. It
throws this error if there is a mismatch: "Please update the PROTOC_VERSION in
your WORKSPACE". Here for search reasons :-).

Tensorflow sets its version of Protobuf in this
[WORKSPACE](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/workspace2.bzl)
file. Reverb sets its version of protoc in this
[WORKSPACE](https://github.com/deepmind/reverb/blob/master/WORKSPACE) file. The
twist is the protobuf people release protoc as 21.9 and the protobuf library as
3.21.0 as seen
[here](https://github.com/protocolbuffers/protobuf/releases/tag/v21.12).

Until Feb-2023, Tensorflow was using a version of Protobuf that was ~2 years
old. In Feb 2023, TF jumped to the latest protobuf 3.21.9.

Note: While I am pretty sure Tensorflow was on 21.9 as of Feb-2023, The checker
in Reverb saw it as 21.0. 21.0 worked so I (tobyboyd) did not look into it any
farther.
