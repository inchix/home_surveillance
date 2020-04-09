FROM nvcr.io/nvidia/l4t-base:r32.3.1
MAINTAINER domcross

# prevent interactive prompts while installing packages, e.g. ssh or tzdata
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libopenblas-base wget git setuptools
RUN apt-get install -y --no-install-recommends libblas-dev liblapack-dev libatlas-base-dev gfortran python3-dev openmpi-bin openmpi-common

RUN python3 -m pip install -U pip

# pytorch 1.4
RUN wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install Cython numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl

# OpenCV 4.2
RUN wget https://raw.githubusercontent.com/domcross/nano_build_opencv/master/build_opencv.sh
RUN bash ./build_opencv.sh

# dlib
RUN git clone https://github.com/davisking/dlib.git
RUN cd dlib && python3 setup.py install --set DLIB_USE_CUDA=1 && cd ..

# openface
RUN git clone https://github.com/cmusatyalab/openface
RUN cd openface && python3 setup.py install && cd ..

# home surveillance system
RUN git clone https://github.com/domcross/home_surveillance.git
RUN cd home_surveillance
RUN python3 -m pip install numpy pandas>=1.0.2 requests>=2.23.0 psutil>=5.7.0 scikit-learn scipy Werkzeug==0.16.1 websocket-client apprise
RUN python3 -m pip install Flask==0.11.1 Flask-Uploads==0.2.1 Flask-SocketIO==2.5
RUN mkdir aligned-images && mkdir training-images

EXPOSE 5000/tcp
