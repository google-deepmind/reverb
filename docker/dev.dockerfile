# Run the following commands in order:
#
# REVERB_DIR="/tmp/reverb"  # (change to the cloned reverb directory, e.g. "$HOME/reverb")
# docker build --tag tensorflow:reverb - < "$REVERB_DIR/docker/dev.dockerfile"
# docker run --rm -it -v ${REVERB_DIR}:/tmp/reverb \
#   -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro \
#   --name reverb tensorflow:reverb bash
#
# Test that everything worked:
#
#
# export PYTHON_BIN_PATH=/usr/bin/python3.7
# $PYTHON_BIN_PATH ./configure.py
# bazel test -c opt --test_output=streamed //reverb:tf_client_test

ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Reverb Team <no-reply@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
ARG tensorflow_pip="tf-nightly"
ARG python_version="/usr/bin/python3.6"
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Pick up some TF dependencies
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
        software-properties-common \
        aria2 \
        build-essential \
        curl \
        gdb \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python3-distutils \
        python3.6-dev \
        python3.7-dev \
        python3.8-dev \
        python3.8-distutils \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py

ARG bazel_version=3.7.2
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Add support for Bazel autocomplete, see
# https://docs.bazel.build/versions/master/completion.html for instructions.
RUN cp /usr/local/lib/bazel/bin/bazel-complete.bash /etc/bash_completion.d
# TODO(b/154932078): This line should go into bashrc.
# NOTE(ebrevdo): RUN source doesn't work.  Disabling the command below for now.
# RUN source /etc/bash_autcompletion.d/bazel-complete.bash

ARG pip_dependencies=' \
      contextlib2 \
      dm-tree>=0.1.5 \
      dataclasses \
      google-api-python-client \
      h5py \
      numpy \
      oauth2client \
      pandas \
      portpicker'


# So dependencies are installed for the supported Python versions
RUN $python_version get-pip.py
RUN $python_version -mpip --no-cache-dir install ${tensorflow_pip} --upgrade
RUN $python_version -mpip --no-cache-dir install $pip_dependencies

RUN rm get-pip.py

# `bazel test` ignores bazelrc and only uses /usr/bin/python3.
# TODO(b/157223742). This also means that regardless of the version of python
# you use for configure.py this is the pythong that will be used by
# `bazel test`. Bazel assumes that the python executable is "python".
RUN rm /usr/bin/python3
RUN ln -s $python_version /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR "/tmp/reverb"

CMD ["/bin/bash"]
