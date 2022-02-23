#!/bin/bash

# default
BUILD_TARGET="mpa-cli" # "mpa-cli" | "torch-base" | "all"
VER_CUDA="11.1"
VER_TORCH="1.8.2"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -b|--build_target)
      BUILD_TARGET="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--push)
      PUSH="yes"
      shift # past argument
      ;;
    -c|--cuda)
      VER_CUDA="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--torch)
      VER_TORCH="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      DEFAULT="yes"
      break
      shift # past argument
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

if [ "$#" -lt 1 ] || [ "$DEFAULT" == "yes" ]; then
cat << EndofMessage
    USAGE: $0 <tag> [Options]

    Options
        -b|--build_target   Specify build target(s). choose one of ["mpa-cli", "torch-base", "all"]. default "mpa-cli"
        -p|--push           Push built image(s) to registry
        -c|--cuda           Specify CUDA version if build_target option is "torch_base" or "all"
        -t|--torch          Specify Pytorch version if build_target option is "torch_base" or "all"
        -h|--help           Print this message
EndofMessage
exit 0
fi

TAG=$1

if (( $(echo "$VER_CUDA >= 11.0" |bc -l) )); then
    VER_CUDNN=8
else
    VER_CUDNN=7
fi

cp -f .dockerignore.bak .dockerignore

TORCH_BASE_TAG="$VER_TORCH-cu$VER_CUDA"

if [ "$BUILD_TARGET" == "all" ] || [ "$BUILD_TARGET" == "torch-base" ]; then
    # build torch-base image
    docker build -f docker/Dockerfile.torch-base \
    --build-arg ver_cuda=$VER_CUDA \
    --build-arg ver_cudnn=$VER_CUDNN \
    --build-arg ver_pytorch=$VER_TORCH \
    --tag torch-base:$TORCH_BASE_TAG ./docker; RET=$?
    if [ $RET -ne 0 ]; then
        echo "failed to build a 'torch-base' image. $RET"
        rm .dockerignore
        exit -1
    fi
    #  build torch-base-test
    docker build -f docker/Dockerfile.torch-base.test \
    --build-arg BASE="torch-base:${TORCH_BASE_TAG}" \
    --tag torch-base-test:$TORCH_BASE_TAG ./docker; RET=$?
    if [ $RET -ne 0 ]; then
        echo "failed to build a 'torch-base-test' image. $RET"
        rm .dockerignore
        exit -1
    fi
fi

docker build -f Dockerfile \
--build-arg BASE=torch-base-test:$TORCH_BASE_TAG \
--tag registry.toolbox.iotg.sclab.intel.com/mpa/cu$VER_CUDA/cli:$TAG \
--tag registry.toolbox.iotg.sclab.intel.com/mpa/cu$VER_CUDA/cli:latest .; RET=$?
if [ $RET -ne 0 ]; then
    echo "failed to build a 'mpa/cu$VER_CUDA/cli' image. $RET"
    rm .dockerignore
    exit -1
fi
rm .dockerignore

echo "Successfully built docker image."

if [ "$PUSH" == "yes" ]; then
    docker push registry.toolbox.iotg.sclab.intel.com/mpa/cu$VER_CUDA/cli:$TAG; RET=$?
    if [ $RET -ne 0 ]; then
        echo "failed to push a docker image to registry. $RET"
        exit -1
    fi
    docker push registry.toolbox.iotg.sclab.intel.com/mpa/cu$VER_CUDA/cli:latest; RET=$?
    if [ $RET -ne 0 ]; then
        echo "failed to push a docker image to registry. $RET"
        exit -1
    fi
else
    echo "Newly built image was not pushed to the registry. use '-p|--push' option to push image."
fi
