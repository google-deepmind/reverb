# How to install Reverb in a docker container

1. Set the `REVERB_DIR` environment variable to point to your Reverb installation.

  ```shell
  $ export REVERB_DIR=/path/to/reverb
  ```

1. Build the docker container.

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

1. Compile Reverb.

  ```shell
  $ bazel build -c opt --config=manylinux2010 //reverb/pip_package:build_pip_package
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
