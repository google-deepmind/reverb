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
# bazel test -c opt --test_output=streamed //reverb:tf_client_test

ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Reverb Team <no-reply@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:18.04"
ARG base_image=$cpu_base_image

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        git \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        python3-distutils \
        python3-dev \
        python3.6-dev \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

ARG bazel_version=2.2.0
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
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
      dm-tree \
      google-api-python-client \
      h5py \
      numpy \
      oauth2client \
      pandas \
      portpicker'

RUN pip3 --no-cache-dir install $pip_dependencies

# The latest tensorflow requires CUDA 10 compatible nvidia drivers (410.xx).
# If you are unable to update your drivers, an alternative is to compile
# tensorflow from source instead of installing from pip.
# Ensure we install the correct version by uninstalling first.
RUN pip3 uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu

RUN pip3 --no-cache-dir install tf-nightly --upgrade

# bazel assumes the python executable is "python".
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR "/tmp/reverb"

CMD ["/bin/bash"]
