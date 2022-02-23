#!/bin/bash
#################################################################
# this script is deprecated one for running unittests
# do not use it for your testing. it could be used only
# to check availability of docker-compose based (as a service)
# mpa execution only.
#################################################################

export UID=$(id -u)
export GID=$(id -g)

#
# create share folders - will be mounted to the MPA container.
#
mkdir -p logs
mkdir -p data

#
# Variables
#
ARTIFACT_DIR="./.reports"
OUTPUT_FILE="unit-tests.xml"
COV_REPORT="none"
BUILD_NAME_SEED="0"
if [ $(pwd | cut -d "/" -f2) == "ci" ]; then
    RUNNER_PREFIX=$(pwd | cut -d "/" -f7)
fi

if [ -z "$TEST_TYPE" ]; then
    TEST_TYPE="auto"
fi

#
# Get arguments via commandline
#

while [[ $# -gt  0 ]];
do
    key="$1"

    case "$key" in
        # -ad|--artifact-dir)
        #     ARTIFACT_DIR=$(realpath "$2")
        #     shift
        #     shift
        #     ;;
        -o|--output_file)
            OUTPUT_FILE=$2
            shift
            shift
            ;;
        -cr|--cov-report)
            COV_REPORT=$2
            shift
            shift
            ;;
        *)
    esac
done

#
# Run tests with docker-compose based on $MODE
#

# if [ -d $REPORT_DIR ] ; then
rm -rf $ARTIFACT_DIR/*
# fi
mkdir -p "$ARTIFACT_DIR"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    PROJECT_NAME="mpa_cpu${BUILD_NAME_SEED}_${TEST_TYPE}"
    echo "Running tests on CPUs"
    YAML_FILES="-p ${PROJECT_NAME} -f docker-compose.ci.cpu.test.yaml"
    WAIT_SECS=5
else
    PROJECT_NAME="mpa_gpu${BUILD_NAME_SEED}_${TEST_TYPE}"
    echo "Running tests on GPUs"
    YAML_FILES="-p ${PROJECT_NAME} -f docker-compose.ci.gpu.test.yaml"
    WAIT_SECS=5
    if [ -z "$VER_CUDA" ]; then
        VER_CUDA="$(nvidia-smi | awk -F"CUDA Version:" 'NR==3{split($2,a," ");print a[1]}')"
        # export VER_CUDA="11.1"
        echo "recommended CUDA version = $VER_CUDA"
    fi
fi

CONTAINER_NAME="cli"

if [ -z "$VER_CUDA" ]; then
    echo "no CUDA version is specified."
    export VER_PYTORCH=1.8.2
    # export VER_TORCHVISION=0.7
else
    if (( $(echo "$VER_CUDA >= 11.0" |bc -l) )); then
        VER_CUDNN=8
        export VER_PYTORCH=1.8.2
        # export VER_TORCHVISION=0.8
    else
        VER_CUDNN=7
        export VER_PYTORCH=1.8.2
        # export VER_TORCHVISION=0.7
    fi

    export CUDA_TAG=$(wget -q https://registry.hub.docker.com/v1/repositories/nvidia/cuda/tags -O -  | \
        sed -e 's/[][]//g' -e 's/"//g' -e 's/ //g' | tr '}' '\n'  | awk -F: '{print $3}' | grep cudnn${VER_CUDNN}-devel-ubuntu18.04 | \
        grep ${VER_CUDA} | awk -F "" 'END{print}')
fi

if [ -z "$TT_UNIT_TESTS" ]; then
    TT_UNIT_TESTS="False"
fi
if [ -z "$TT_COMPONENT_TESTS" ]; then
    TT_COMPONENT_TESTS="False"
fi

if [ $TT_UNIT_TESTS == "False" ] && [ $TT_COMPONENT_TESTS == "False" ]; then
    TT_UNIT_TESTS="True"
fi

cp -f .dockerignore.bak .dockerignore
docker-compose ${YAML_FILES} up --build -d; RET=$?
rm .dockerignore

if [ $RET -ne 0 ]; then
    echo "failed to build and start application"
    unset GID
    unset VER_PYTORCH
    # unset VER_TORCHVISION
    unset CUDA_TAG
    exit -1
fi

echo "Waiting for the application totally up..."
sleep $WAIT_SECS

REPORT_OPTS="-s -v --junitxml=$ARTIFACT_DIR/$OUTPUT_FILE"
if [ $COV_REPORT != "none" ]; then
    REPORT_OPTS="$REPORT_OPTS --cov-report ${COV_REPORT}:${ARTIFACT_DIR}/htmlcov"
fi

if [ $TEST_TYPE == "nightly" ]; then
    REPORT_OPTS="$REPORT_OPTS tests/intg"
else
    REPORT_OPTS="$REPORT_OPTS --cov mpa tests/unit"
fi

docker-compose ${YAML_FILES} exec -T \
    --env TT_COMPONENT_TESTS=${TT_COMPONENT_TESTS} \
    --env TT_UNIT_TESTS=${TT_UNIT_TESTS} \
    $CONTAINER_NAME /bin/bash -c "python -m pytest ${REPORT_OPTS}"; RET=$?

if [ $RET -ne 0 ]; then
    echo "unittest exits with non-zero."
    echo "------ LOGS FOR ${CONTAINER_NAME} ------"
    docker logs ${PROJECT_NAME}_${CONTAINER_NAME}_1
fi
# copy pytest result to artifact dir
# docker cp ${PROJECT_NAME}_${CONTAINER_NAME}_1:$REPORT_DIR/$OUTPUT_FILE $ARTIFACT_DIR; echo "[Ret code for copying pytest results: ${?}]"

docker-compose ${YAML_FILES} down; echo "[Ret code for service stopping: ${?}]"

if [ $RET -ne 0 ]; then
    echo "test:FAILED exits with non-zero."
    unset GID
    unset VER_PYTORCH
    # unset VER_TORCHVISION
    unset CUDA_TAG
    exit -1
fi

