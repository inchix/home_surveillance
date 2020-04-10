#FROM nvcr.io/nvidia/cuda:10.2-runtime
FROM nvcr.io/nvidia/pytorch:20.03-py3
MAINTAINER inchix

# prevent interactive prompts while installing packages, e.g. ssh or tzdata
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y --no-install-recommends apt-utils python3-pip libopenblas-base wget git python3-setuptools build-essential libblas-dev liblapack-dev libatlas-base-dev gfortran python3-dev openmpi-bin openmpi-common libfreetype6-dev

RUN python3 -m pip install -U pip

# pytorch 1.4
#RUN wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install Cython
#RUN python3 -m pip install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install  matplotlib pandas>=1.0.2 requests>=2.23.0 psutil>=5.7.0 scikit-learn scipy Werkzeug==0.16.1 websocket-client apprise Flask==0.11.1 Flask-Uploads==0.2.1 Flask-SocketIO==2.5

# OpenCV 4.2
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends build-essential \
        cmake \
        git \
        gfortran \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libcanberra-gtk3-module \
        libdc1394-22-dev \
        libeigen3-dev \
        libglew-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libjpeg8-dev \
        libjpeg-turbo8-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        libpng-dev \
        libpostproc-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtesseract-dev \
        libtiff-dev \
        libv4l-dev \
        libxine2-dev \
        libxvidcore-dev \
        libx264-dev \
        pkg-config \
        python3-dev \
        qv4l2 \
        v4l-utils \
        v4l2ucp \
        zlib1g-dev
RUN cd /tmp && mkdir build_opencv && cd build_opencv && \
        git clone --branch 4.2.0 https://github.com/opencv/opencv.git && git clone --branch 4.2.0 https://github.com/opencv/opencv_contrib.git && \
        cd opencv && mkdir build && cd build && \
        cmake -D BUILD_EXAMPLES=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=RELEASE -D CUDA_ARCH_BIN=5.3 -D CUDA_ARCH_PTX= -D CUDA_FAST_MATH=ON -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=/tmp/build_opencv/opencv_contrib/modules -D OPENCV_GENERATE_PKGCONFIG=ON -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D WITH_OPENGL=ON .. && \ 
        #cmake -D BUILD_EXAMPLES=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=RELEASE -D CUDA_ARCH_BIN=5.3 -D CUDA_ARCH_PTX= -D CUDA_FAST_MATH=ON -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 -D ENABLE_NEON=ON -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=/tmp/build_opencv/opencv_contrib/modules -D OPENCV_GENERATE_PKGCONFIG=ON -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D WITH_OPENGL=ON .. && \ 
        make -j 1 && \ 
        make install && \
        rm -rf /tmp/build_opencv
#RUN wget https://raw.githubusercontent.com/mdegans/nano_build_opencv/master/build_opencv.sh && bash ./build_opencv.sh

# dlib
RUN git clone https://github.com/davisking/dlib.git && cd dlib && python3 setup.py install --set DLIB_USE_CUDA=1 && cd ..

# openface
RUN git clone https://github.com/cmusatyalab/openface && cd openface && python3 setup.py install && cd ..

# home surveillance system
RUN git clone https://github.com/domcross/home_surveillance.git && cd home_surveillance && mkdir aligned-images && mkdir training-images

EXPOSE 5000/tcp
