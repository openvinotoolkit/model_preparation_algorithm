#!/bin/bash
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    VER_CUDA="10.2"
    RUNTIME_ARG=""
    ENV_ARG=""
else
    if [ -z "$VER_CUDA" ]; then
        VER_CUDA="$(nvidia-smi | awk -F"CUDA Version:" 'NR==3{split($2,a," ");print a[1]}')"
    fi
    RUNTIME_ARG="--runtime=nvidia"
    ENV_ARG="--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# pull latest image from registry if there is updates
if [ -z "$SKIP_PULL" ]; then
    docker pull registry.toolbox.iotg.sclab.intel.com/mpa/cu${VER_CUDA}/cli:latest
fi

echo "*" > .dockerignore
export GID=$(id -g)
docker build -f Dockerfile.dev \
--build-arg gid=$GID \
--build-arg uid=$UID \
--build-arg VER_CUDA=${VER_CUDA} \
--tag mpa-cli-dev:latest .; RET=$?
rm .dockerignore

if [ $RET -ne 0 ]; then
    echo "failed to build mpa-cli-dev image"
    unset GID
    exit -1
fi 

# create folders to be mounted to mpa/cli container if not exists
mkdir -p logs
mkdir -p data
mkdir -p .reports

if [ -z "$TT_UNIT_TESTS" ]; then
    TT_UNIT_TESTS="False"
fi
if [ -z "$TT_COMPONENT_TESTS" ]; then
    TT_COMPONENT_TESTS="False"
fi

if [ $TT_UNIT_TESTS == "False" ] && [ $TT_COMPONENT_TESTS == "False" ]; then
    TT_UNIT_TESTS="True"
fi

MPA_ARGS="$@"
echo "run with args: ${MPA_ARGS}"
docker run -t --rm ${RUNTIME_ARG} ${ENV_ARG} \
--env TT_COMPONENT_TESTS=${TT_COMPONENT_TESTS} \
--env TT_UNIT_TESTS=${TT_UNIT_TESTS} \
--shm-size 5g \
-v $(pwd)/data:/usr/src/app/data \
-v $(pwd)/models:/usr/src/app/models \
-v $(pwd)/recipes:/usr/src/app/recipes \
-v $(pwd)/recipes_old:/usr/src/app/recipes_old \
-v $(pwd)/samples:/usr/src/app/samples \
-v $(pwd)/mpa:/usr/src/app/mpa \
-v $(pwd)/tools:/usr/src/app/tools \
-v $(pwd)/tests:/usr/src/app/tests \
-v $(pwd)/logs:/usr/src/app/logs \
-v $(pwd)/.reports:/usr/src/app/.reports \
-u mpa \
mpa-cli-dev:latest /bin/bash -c "python -m $MPA_ARGS"
