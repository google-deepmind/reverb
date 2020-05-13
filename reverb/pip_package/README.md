# How to develop and build Reverb with the Docker containers

## Overview

This document covers a couple scenarios:

 *  <a href='#Release'>Create a Reverb release</a>
 *  <a href='#Develop'>Develop Reverb inside a docker container</a>


While there is overlap in the uses cases, treating them as separately with
some copy and pasted commands seems the most clear at the moment. Before you
get started setup a local variable pointing to your local Reverb github repo.

```shell
$ export REVERB_DIR=/path/to/reverb/github/repo
```

<a id='Release'></a>
## Create a Reverb release

These are the steps to create a Reverb release wheel using the Reverb
release.dockerfile. This is part of the process used to create official Reverb
releases. The other parts are creating branches, tags, and uploading to pypi.

Execute from the root of the git repository. The end result will end up in
`$REVERB_DIR/dist`.

```shell
# By default the container is configured with Python 3.6.
$ docker build --tag tensorflow:reverb_release \
  - < "$REVERB_DIR/docker/release.dockerfile"

$ docker run --rm -it --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
tensorflow:reverb_release bash oss_build.sh --python 3.6


# Alternatively configure and build for Python 3.8.
$ docker build --tag tensorflow:reverb --build-arg python_version=python3.8 \
  - < "$REVERB_DIR/docker/dev.dockerfile"

$ docker run --rm -it --mount "type=bind,src=$REVERB_DIR,dst=/tmp/reverb" \
tensorflow:reverb_release bash oss_build.sh --python 3.8

```

<a id='Develop'></a>
## Develop Reverb inside a docker container


1. Build the docker container. By default the container is setup for python 3.6.
   Use the `python_version` arg to configure the container with 3.7, 3.8, or
   all versions with
   `--build-arg python_version="python3.6 python3.7 python3.8"`.

  ```shell
$ docker build --tag tensorflow:reverb - < "$REVERB_DIR/docker/dev.dockerfile"

# Alternatively you can build the container with Python 3.7 support.
$ docker build --tag tensorflow:reverb --build-arg python_version=python3.7 \
  - < "$REVERB_DIR/docker/dev.dockerfile"
  ```

1. Run and enter the docker container.

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
bazel build -c opt //reverb/pip_package:build_pip_package
  ```

1. Build the .whl file and output it to `/tmp/reverb_build/dist/`.

  ```shell
$ ./bazel-bin/reverb/pip_package/build_pip_package --dst /tmp/reverb_build/dist/
  ```

1. Install the .whl file.

  ```shell
# If multiple versions were built, pass the exact wheel to install rather than
# *.whl.
$ $PYTHON_BIN_PATH -mpip install --upgrade /tmp/reverb_build/dist/*.whl
  ```





