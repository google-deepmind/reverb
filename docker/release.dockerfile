# Run the following commands in order:
#
# REVERB_DIR="/tmp/reverb"  # (change to the cloned reverb directory, e.g. "$HOME/reverb")
# docker build --tag tensorflow:reverb_release - < "$REVERB_DIR/docker/release.dockerfile"
# docker run --rm -it -v ${REVERB_DIR}:/tmp/reverb \
#   -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro \
#   --name reverb_release tensorflow:reverb_release bash
#
# Test that everything worked:
#
# bazel test -c opt --copt=-mavx --config=manylinux2010 --test_output=errors //reverb/...

ARG cpu_base_image="tensorflow/tensorflow:2.1.0-custom-op-ubuntu16"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Reverb Team <no-reply@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="tensorflow/tensorflow:2.1.0-custom-op-ubuntu16"
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
        python3-dev \
        python3.6-dev \
        python3.7-dev \
        python3.8-dev \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

ARG pip_dependencies=' \
      absl-py \
      contextlib2 \
      dm-tree>=0.1.5 \
      google-api-python-client \
      h5py \
      numpy \
      oauth2client \
      pandas \
      portpicker'

# TODO(b/154930404): Update to 2.2.0 once it's out.  May need to
# cut a branch to make changes that allow us to build against 2.2.0 instead
# of tf-nightly due to API changes.
RUN pip3 uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu
RUN pip3 --no-cache-dir install tf-nightly --upgrade

RUN pip3 --no-cache-dir install $pip_dependencies

WORKDIR "/tmp/reverb"

CMD ["/bin/bash"]
