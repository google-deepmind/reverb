# How to develop and build Reverb with the Docker containers

## Overview

This document covers a couple scenarios:

 *  <a href='#Release'>Create a Reverb release</a>
 *  <a href='#Develop'>Develop Reverb inside a Docker container</a>


While there is overlap in the uses cases, treating them as separately with
some copy and pasted commands seems the most clear at the moment. Before you
get started setup a local variable pointing to your local Reverb github repo.

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
# Builds the container with Python 3.6, 3.7, 3.8, and 3.9. Set the
# `build-arg tensorflow_pip` to the version of TensorFlow to build against.
$ docker build --tag tensorflow:reverb_release \
  --build-arg tensorflow_pip=tensorflow~=2.5.0 \
  --build-arg python_version="python3.6 python3.7 python3.8 python3.9" \
  - < "$REVERB_DIR/docker/release.dockerfile"

#################################################
# Builds Reverb against TensorFlow stable or rc.
#################################################

# Builds Reverb against most recent stable release of TensorFlow and
# requires `tensorflow~=2.5.0` if using `pip install reverb[tensorflow]`.
# Packages for Python 3.6, 3.7, 3.8, and 3.9 are created.
$ docker run --rm --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
  tensorflow:reverb_release bash oss_build.sh --clean true \
  --tf_dep_override "tensorflow~=2.5.0" --release --python "3.6 3.7 3.8 3.9"

# Builds Reverb against an RC of TensorFlow. `>=` and `~=` are not effective
# because pip does not recognize 2.4.0rc0 as greater than 2.3.0. RC builds need
# to have a strict dependency on the RC of TensorFlow used.
$ docker run --rm --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
  tensorflow:reverb_release bash oss_build.sh --clean true \
  --tf_dep_override "tensorflow==2.4.0rc0" --release --python "3.6 3.7 3.8 3.9"
```

<a id='Develop'></a>
## Develop Reverb inside a Docker container

1. Build the Docker container. By default the container is setup for python 3.6.
   Use the `python_version` arg to configure the container with 3.7, 3.8, or
   all versions with
   `--build-arg python_version="python3.6 python3.7 python3.8"`.

  ```shell
  $ docker build --tag tensorflow:reverb - < "$REVERB_DIR/docker/dev.dockerfile"

  # Alternatively you can build the container with Python 3.7 support.
  $ docker build --tag tensorflow:reverb --build-arg python_version=python3.7 \
    - < "$REVERB_DIR/docker/dev.dockerfile"
  ```

1. Run and enter the Docker container.

  ```shell
  $ docker run --rm -it \
    --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
    --mount "type=bind,src=$HOME/.gitconfig,dst=/etc/gitconfig,ro" \
    --name reverb tensorflow:reverb bash
  ```

1. Define the Python version to use (python3, python3.7, python3.8), which needs
   the match the versions of python you built the container to support.
   `python3` is equivalent to python3.6 and is the default python in the
   container.

  ```shell
  $ export PYTHON_BIN_PATH=python3
  ```

1. Compile Reverb.

  ```shell
  $ $PYTHON_BIN_PATH configure.py
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
