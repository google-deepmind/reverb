ARG cpu_base_image="tensorflow/build:latest-python3.7"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="no-reply@google.com"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="tensorflow/build:latest-python3.7"
ARG base_image=$cpu_base_image
ARG tensorflow_pip="tf-nightly"
ARG python_version="python3.7"
ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

# Stops tzdata from asking about timezones and blocking install on user input.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Pick up some TF dependencies
RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
        software-properties-common \
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
        python3.7-dev \
        python3.8-dev \
        python3.9-dev \
        python3.10-dev \
        # python >= 3.8 needs distutils for packaging.
        python3.8-distutils \
        python3.9-distutils \
        python3.10-distutils \
        rename \
        rsync \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py

# Installs known working version of bazel.
ARG bazel_version=3.7.2
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

ARG pip_dependencies=' \
      absl-py \
      contextlib2 \
      dataclasses \
      dm-tree>=0.1.5 \
      google-api-python-client \
      h5py \
      mock \
      numpy \
      oauth2client \
      pandas \
      portpicker'

RUN for python in ${python_version}; do \
    $python get-pip.py && \
    $python -mpip uninstall -y tensorflow tensorflow-gpu tf-nightly tf-nightly-gpu && \
    $python -mpip --no-cache-dir install ${tensorflow_pip} --upgrade && \
    $python -mpip --no-cache-dir install $pip_dependencies; \
  done
RUN rm get-pip.py

# Removes existing links so they can be created to point where we expect.
RUN rm /dt7/usr/include/x86_64-linux-gnu/python3.8
RUN rm /dt7/usr/include/x86_64-linux-gnu/python3.9
RUN rm /dt7/usr/include/x86_64-linux-gnu/python3.10

# Needed until this is included in the base TF image.
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.8" "/dt7/usr/include/x86_64-linux-gnu/python3.8"
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.9" "/dt7/usr/include/x86_64-linux-gnu/python3.9"
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.10" "/dt7/usr/include/x86_64-linux-gnu/python3.10"

CMD ["/bin/bash"]
