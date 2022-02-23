# How to build base docker image for MPA
You can build docker image by following instructions. Target version of some modules / components below should be changed for your environment.
## Create pytorch base image and tag it as base image
```bash
$ docker build -f docker/Dockerfile.torch-base --build-arg ver_cuda="10.2" --build-arg ver_cudnn="7" --build-arg ver_pytorch="1.7.1" --tag torch-base:1.7.1-cu10.2 docker
```
## Create test image with iotg e2e framework
```bash
$ docker build -f docker/Dockerfile.torch-base.test --build-arg BASE=torch-base:1.7.1-cu10.2 --tag torch-base-test:1.7.1-cu10.2 docker
```

## Create MPA docker image
* build image using [Dockerfile](mpa/Dockerfile)
```bash
$ docker build -f Dockerfile --build-arg BASE=torch-base-test:1.7.1-cu10.2 --tag mpa/cu10.2/cli:1.0.0 .
```

## Tag and push docker image
```bash
$ docker tag mpa/cu10.2/cli:1.0.0 registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli:1.0.0
$ docker tag registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli:1.0.0 \
registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli:latest
$ docker push registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli:1.0.0
$ docker push registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli:latest
```