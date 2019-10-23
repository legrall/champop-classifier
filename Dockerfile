FROM tensorflow/tensorflow:1.14.0-py3-jupyter


WORKDIR /opt

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y git wget


# Install fork of Mask-RCNN
RUN git clone https://github.com/allan-ja/Mask_RCNN.git &&\
    cd Mask_RCNN &&\
    pip install -r requirements.txt &&\
    python setup.py install

RUN mkdir -p /opt/data &&\
    wget --quiet https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz &&\
    tar xzf dtd-r1.0.1.tar.gz -C /opt/data &&\
    rm dtd-r1.0.1.tar.gz

RUN wget --quiet -P /opt/data https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5


WORKDIR /opt/champop

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .


CMD ["python", "champop.py", "train"]