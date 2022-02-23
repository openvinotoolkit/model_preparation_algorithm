ARG BASE
FROM ${BASE}

WORKDIR /usr/src/app

# install mmdetection
ENV FORCE_CUDA="1"
COPY ./external  /usr/src/app/external

# prerequisite for the mmdetection
RUN apt-get update && apt-get install -y git && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN cd ./external/mmdetection/ && \
    pip install -r requirements/build.txt && \
    pip install -v -e . && \
    pip install -r requirements/runtime.txt && \
    cd -

# install mmsegmentation
RUN cd ./external/mmsegmentation/ && \
    pip install -r requirements && \
    pip install -v -e . && \
    cd -

# install MDA
RUN pip install -v -e ./external/mda

# install HPO
RUN pip install -v -e ./external/hpo

# install ote-sdk
RUN pip install -v -e ./external/training_extensions/ote_sdk

COPY ./requirements.txt .
RUN pip install -r ./requirements.txt

# copying unittest stuffs
COPY ./pytest.ini .
COPY ./.coveragerc .
