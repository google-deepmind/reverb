# How to install Reverb in a docker container

1. Set the `REVERB_DIR` environment variable to point to your Reverb installation.

  ```shell
  $ export REVERB_DIR=/path/to/reverb
  ```

1. Build the docker container (release.dockerfile for manylinux2010 or
  dev.dockerfile with Ubuntu18.04 and python3.8 support).

  ```shell
  $ docker build --tag tensorflow:reverb - < "$REVERB_DIR/docker/release.dockerfile"
  ```

1. Run the docker container.

  ```shell
  $ docker run --rm -it \
    --mount "type=bind,src=$REVERB_DIR,dst=$REVERB_DIR" \
    --mount "type=bind,src=$HOME/.gitconfig,dst=/etc/gitconfig,ro" \
    --name reverb tensorflow:reverb \
    bash
  ```

1. (Optional) Define the Python version to use (by default, python3.6 will be
    used).

  ```shell
  $ export PYTHON_BIN_PATH=python3.{X}
  ```

  Where {X} is 6, 7 or 8.

1. Compile Reverb.
    * Option a: if using the release.dockerfile and overwriting the python version:

  ```shell
  $ bazel build -c opt --config=manylinux2010 \
    --action_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH \
    --repo_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH \
    --python_path=$PYTHON_BIN_PATH \
    //reverb/pip_package:build_pip_package
  ```
  * Option b: if using the release.dockerfile with the default python:

  ```shell
  $ bazel build -c opt --config=manylinux2010 \
    //reverb/pip_package:build_pip_package
  ```

 * Option c: if using the dev.dockerfile and overwriting the python version:

  ```shell
  $ bazel build -c opt \
    --action_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH \
    --repo_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH \
    --python_path=$PYTHON_BIN_PATH \
    //reverb/pip_package:build_pip_package
  ```
  * Option d: if using the dev.dockerfile with the default python:

  ```shell
  $ bazel build -c opt \
    //reverb/pip_package:build_pip_package
  ```

1. Build the .whl file. This will create a temporary directory, copy source files from bazel-bin and use Python's distutils to create a whl file.

  ```shell
  $ ./reverb/pip_package/build_pip_package.sh
  ```

1. Copy the name of the temporary direcotry that was created (it is printed
to stdout during the previous step).

1. Install the .whl file.

  ```shell
  $ pip install <TMP_DIR>/dist/<NAME_OF_WHL_FILE>.whl
  ```
