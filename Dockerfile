FROM ubuntu:22.04
WORKDIR /slam
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qqy update \
    && apt-get -qqy upgrade \
    && apt-get -qqy install --no-install-recommends build-essential \
    && apt-get -qqy install --no-install-recommends g++ \
    && apt-get -qqy install --no-install-recommends gdb \
    && apt-get -qqy install --no-install-recommends gdbserver \
    && apt-get -qqy install --no-install-recommends cmake \
    && apt-get -qqy install --no-install-recommends git \
    && apt-get -qqy install --no-install-recommends ca-certificates
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qqy update \
    && apt-get -qqy upgrade \
    && apt-get -qqy install --no-install-recommends libeigen3-dev \
    && apt-get -qqy install --no-install-recommends libopencv-dev \
    && apt-get -qqy install --no-install-recommends libglfw3-dev \
    && apt-get -qqy install --no-install-recommends libgl-dev \
    && apt-get -qqy install --no-install-recommends libglu-dev
RUN mkdir -p /dependencies \
    && cd /dependencies \
    && git clone https://github.com/RainerKuemmerle/g2o.git \
    && cd /dependencies/g2o \
    && mkdir build \
    && cd /dependencies/g2o/build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && cmake --build . --config Release --parallel 4 \
    && cmake --build . --config Release --parallel 4 --target install
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qqy update \
    && apt-get -qqy upgrade \
    && apt-get -qqy install --no-install-recommends libcanberra-gtk-module \
    && apt-get -qqy install --no-install-recommends libcanberra-gtk3-module
RUN ldconfig
ENV NO_AT_BRIDGE=1
CMD ["bash"]
