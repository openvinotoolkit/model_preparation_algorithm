#!/bin/bash
#
# INTEL CONFIDENTIAL
# Copyright (c) 2020 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
#


#################################################################################
# run tests
#################################################################################
OS=`uname`

echo "BUILD_NAME_SEED=${BUILD_NAME_SEED}"
echo "MPA_IMAGE_NAME=${MPA_IMAGE_NAME}"
echo "MPA_IMAGE_TAG=${MPA_IMAGE_TAG}"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $PROJECT_DIR

if [[ $TT_XDIST_WORKERS ]]; then
  XDIST_PARAMETERS="-n ${TT_XDIST_WORKERS}"
else
  XDIST_PARAMETERS=""
fi

if [ -n "$JUNITFILE" ]; then
    # if this is set to true, use default location
    if [ "$JUNITFILE" = "true" ]; then
      JUNITXML="--junitxml=../test_log/tests.xml"
    else
      JUNITXML="--junitxml=${JUNITFILE}"
    fi
    echo "JUnit result file will be generated at '${JUNITXML}'!"
fi

if [ $TT_TEST_TYPE == "CPU" ]; then
    echo "Running tests on CPUs"
    echo "PROJECT_NAME=mpa_project_cpu_name_${BUILD_NAME_SEED}"
    PROJECT_NAME="mpa_project_cpu_name_${BUILD_NAME_SEED}"
    echo "CONTAINER_NAME=mpa_cpu_image${BUILD_NAME_SEED}"
    CONTAINER_NAME="mpa_cpu_image_${BUILD_NAME_SEED}"
    YAML_FILES="-p ${PROJECT_NAME} -f docker-compose.cpu.test.yaml --compatibility"
    WAIT_SECS=10
fi

if [ $TT_TEST_TYPE == "GPU" ]; then
    echo "Running tests on GPUs"
    echo "PROJECT_NAME=mpa_project_gpu_name_${BUILD_NAME_SEED}"
    PROJECT_NAME="mpa_project_gpu_name_${BUILD_NAME_SEED}"
    echo "CONTAINER_NAME=mpa_gpu_image${BUILD_NAME_SEED}"
    CONTAINER_NAME="mpa_gpu_image_${BUILD_NAME_SEED}"
    YAML_FILES="-p ${PROJECT_NAME} -f docker-compose.gpu.test.yaml"
    WAIT_SECS=60
fi

echo "Start docker-compose"
echo "COMMAND: docker-compose ${YAML_FILES} up -d --no-recreate"
docker-compose ${YAML_FILES} up -d --no-recreate; RET=$?
if [ $RET -ne 0 ]; then
    echo "failed to build and start application"
    unset GID
    unset VER_PYTORCH
    unset VER_TORCHVISION
    unset CUDA_TAG
    exit -1
fi

echo "Waiting for the application totally up..."
sleep $WAIT_SECS

echo "Start executing tests"
COVERAGE_FOLDER="/usr/src/app/logs/coverage"
echo "COMMAND: docker exec --env-file <(env | grep -i TT_) ${CONTAINER_NAME} /bin/bash -c "mkdir -p ${COVERAGE_FOLDER} ""
docker exec --env-file <(env | grep -i TT_) ${CONTAINER_NAME} /bin/bash -c "mkdir -p ${COVERAGE_FOLDER}"; RET=$?

echo "COMMAND: docker exec --env-file <(env | grep -i TT_) ${CONTAINER_NAME} /bin/bash -c "python -m pytest -s -v --cov-report html:${COVERAGE_FOLDER} --cov=mpa ${TT_TESTS_DIR} ${JUNITXML} ${XDIST_PARAMETERS}""
docker exec --env-file <(env | grep -i TT_) ${CONTAINER_NAME} /bin/bash -c "python -m pytest -s -v --cov-report html:${COVERAGE_FOLDER} --cov=mpa ${TT_TESTS_DIR} ${JUNITXML} ${XDIST_PARAMETERS}"; RET=$?

if [ $RET -ne 0 ]; then
    echo "unittest exits with non-zero."
    echo "------ LOGS FOR ${CONTAINER_NAME} ------"
    docker logs $CONTAINER_NAME
    exit -1
fi

echo "Stop executing tests"
echo "COMMAND: docker-compose ${YAML_FILES} down"
docker-compose ${YAML_FILES} down; echo "[Ret code for service stopping: ${?}]"
if [ $RET -ne 0 ]; then
    echo "test:FAILED exits with non-zero."
    unset GID
    unset VER_PYTORCH
    unset VER_TORCHVISION
    unset CUDA_TAG
    exit -1
fi

echo "Finished executing tests"
echo "Quiting run_tests.sh"
exit $ret_code
