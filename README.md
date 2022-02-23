# Model Preparation Algorithm

## Introduction
**Model Preparation** is a stage in DL developer's workflow in which
users could create or re-train models based on given dataset.
This package provides various types of Model Preparation Algorithm (MPA)
as the form of **Transfer Learning Recipe** (TL-Recipe).

### TL-Recipe
Given datasets and base models as inputs, TL-Recipe offers model adaptation
schemes in terms of data domain, model tasks and also NN architectures.

This package consists of
* Custom core modules which implement various TL algorithms
* Training configurations specifying module selection & hyper-parameters
* Tools and APIs to run transfer learning on given datasets & models

### Design Concept
>  **TL ( RECIPE**, MODEL, DATA[, H-PARAMS] ) -> MODEL
* Model: NN arch + params to be transfer-learned
    * Explicitly enabled models only (directly or via OMZ plugin)
    * No model input -> train default model from scratch
* Recipe: Custom modules & configs defining transfer learning schemes / modes
    * Defined up to NN arch (e.g. *Incremental learning recipe for FasterRCNN, SSD*, etc)
    * Expose & accept controllable hyper-params w/ default value

### Archtecture
TL-Recipe is based on [pytorch](pytorch.org) as base training framework.
And [MMCV](https://github.com/open-mmlab/mmcv) is adopted as modular configuration framework.
Recipes in this package are implemented using config & module libraries which are also based on MMCV
* [MMClassification](https://github.com/open-mmlab/mmclassification)
* [MMDetection (OTE version)](https://github.com/openvinotoolkit/mmdetection)
* [MMSegmentation (OTE version)](https://github.com/openvinotoolkit/mmsegmentation)

On top of above framework and libraries, we designed **Multi-stage Workflow Framework**
* Workflow: Model training w/ series of sub-stages (RECIPE)
* Stage: Individual (pre)train tasks (SUB-RECIPE)
<img src="doc/workflow-arch.png" alt="worflow-arch" width="800"/>

## Features
TL-Recipe supports various TL *method*s for Computer Vision *task*s and their
related NN *model* architecures.

### TL Tasks
* Classification
* Detection
* (WIP) Re-Identification
* (WIP) Segmentation
    * (WIP) Instance segmentation
    * (TBD) Semantic segmentation
    * (TBD) Panoptic segmentation
> Train / Infer / Evaluate / Export operations are supported for each task

### TL Methods
Follwing table describes supported TL methods for each tasks.

| TL | Type | Classification | Detection | Segmentation |
| -- | ---- | -------------- | --------- | ------------ |
| Domain adaptation | Semi-supervised Learning | FixMatch | STAC | |
| | Self-supervised learning | BYOL | | |
| Task adaptation | Task customize | REPLACE/ADD | REPLACE/ADD/MERGE | |
| | Incremental learning | Task/Class LwF | Class LwF | |
| Arch adaptation | Knowledge Distillation | | | |

### TL Models
TL-Recipe supports transfer learning for subset of
* OTE ([OpenVINO Traning Exetension](https://github.com/openvinotoolkit/training_extensions)) models
* OMZ ([OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)) models
* Some custom pytorch models

For detailed list, please refer to: *TBD*

### Interface
Users could use TL-Recipe to retrain their models via
* Model templates
* [CLI](#how-to-run-cli)
* Python APIs

### Changelog
Please refer to: TBD

## Developer's Guide

### System Requirement
* Ubuntu 16.04 / 18.04 / 20.04 LTS x64
* Intel Core/Xeon
* (optional) NVIDIA GPUs
    * tested with
        * GTX 1080Ti
        * RTX 3080
        * RTX 3090
        * Tesla V100


### Prerequisites
To use MPA on your system, there are two different way to setup the environment for the development.
* Setup environment on your host machine using virtual environments (`virtualenv` or `conda`)
* Setup environment using `Docker`

### For the contributors
To contribute your code to the MPA project, there is a restriction to do liniting to all your changes. we are using Flake8 for doing this and it's recommended to do it using the `Git hook` before commiting your changes.
* how to install: please go to the link below to install on your development environment
    > https://pre-commit.com/#install
* For example, you can install `Flake8` and `pre-commit` using `Anaconda`
    ```bash
    # install flake8 to your development environment (Anaconda)
    $ conda install flake8=3.9 -y
    # install pre-commit module
    $ conda install pre-commit -c conda-forge -y
    # install pre-commit hook to the local git repo
    $ pre-commit install
    ```
    >*__note__*: After installing and setting `Flake8` using instructions above, your commit action will be ignored when your changes (staged files) have any LINT issues.

#### __Setup environment on your host machine__

We provided a shell script `init_venv.sh` to make virtual environment for the MPA using tool `virtualenv`.
```bash
# init submodules
...mpa$ git submodule update --init --recursive
# create virtual environment to the path ./venv and named as 'mpa'
...mpa$ ./init_venv.sh ./venv
# activate created virtualenv
...mpa$ source ./venv/bin/activate
# run mpa cli to get help message
(mpa)...mpa$ python -m tools.cli -h
```

>*__note__*: This script assume that your system has installed suitable version of CUDA (10.2 or 11.1) for the MPA.

Another way to make virtual environment on your machine is that use `Anaconda(miniconda)`. If you want to use the `Anaconda` for virtual environment setting, follow instructions below.
1. Create a `conda` environment and activate it
    ```bash
    # create conda env named as 'mpa'
    $ conda create -n mpa python=3.8
    # activate 'mpa' env
    $ conda activate mpa
    ```

1. install `Pytorch`

    * For CPU only system
        ```bash
        (mpa) .../mpa$ conda install pytorch==1.8.2 torchvision cpuonly -c pytorch-lts -y
        ```
    * For using NVIDIA GPU
        >*__note__*: The latest version of MPA only works with Pytorch 1.8.2 (LTS). and it supports CUDA 10.2 and CUDA 11.1 depending on your system's NVIDIA driver version.

        ```bash
        # example to install torch 1.8.2 and CUDA toolkit
        (mpa) .../mpa$ conda install pytorch==1.8.2 torchvision cudatoolkit=10.2 -c pytorch-lts -y
        # if you want to use CUDA 11.1, you need to add 'nvidia' channel for that
        (mpa) .../mpa$ conda install pytorch==1.8.2 torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
        ```
        >*__note__*: If you want to use NVIDIA GPUs for the training backend, need to specify proper version of the CUDA toolkit to the conda installation argument above. For more details, visit [Pytorch](https://pytorch.org/) web site.
1. install mmcv module
    * For CPU only system
        ```bash
        $ pip install --no-cache-dir mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.2/index.html -c constraints.txt
        ```
    * For using NVIDIA GPU (with CUDA 11.1)
        ```bash
        $ pip install --no-cache-dir mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu11.1/torch1.8.2/index.html -c constraints.txt
        ```
        >*__note__*: For the system which has RTX 30 series GPU and CUDA 11, mmcv installation can be failed. Set `MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80'` to avoid installation issue. Here is the offical workaround from the openmmlab: https://github.com/open-mmlab/mmdetection/blob/master/docs/faq.md#pytorchcuda-environment

        >*__note__*: MPA has dependency with mmdetection and mmsegmentation and both of them require mmcv. so, we need to install proper version of mmcv to satisfy both dependency module. You can check the requirements each of them from files below.
        >* [external/mmdetection/mmdet/\__init__.py](./external/mmdetection/mmdet/__init__.py)
        >* [external/mmsegmentation/mmseg/\__init__.py](./external/mmsegmentation/mmseg/__init__.py)

1. install module dependencies ( `mmcls` will be installed by this command)
    ```base
    (mpa) .../mpa$ pip install -r requirements.txt -c constraints.txt
    ```

1. uninstall `pycocotools`
    ```bash
    (mpa) .../mpa$ pip uninstall pycocotools -y
    ```

1. install `mmpycocotools` and `Polygon3` which has dependency with installed `numpy` version
    ```bash
    (mpa) .../mpa$ pip install --no-cache-dir --no-binary=mmpycocotools mmpycocotools -c constraints.txt
    (mpa) .../mpa$ pip install --no-cache-dir --no-binary=Polygon3 Polygon3==3.0.8 -c constraints.txt
    ```

1. update external modules
    ```bash
    (mpa) .../mpa$ git submodule update --init
    ```

1. install mmdetection (ote)
    ```bash
    (mpa) .../mpa$ cd external/mmdetection && pip install -r requirements.txt && \
    pip install -v -e . && cd -
    ```

1. install mmsegmentation (ote)
    ```bash
    (mpa) .../mpa$ cd external/mmsegmentation && pip install -r requirements.txt && \
    pip install -v -e . && cd -
    ```

1. install MDA
    ```bash
    (mpa) .../mpa$ pip install -v -e external/mda
    ```

1. install HPO module
    ```bash
    (mpa) .../mpa$ pip install -v -e external/hpo
    ```

1. install OTE-SDK
    ```bash
    # install OTE-SDK
    (mpa) .../mpa$ pip install -v -e external/training_extensions/ote_sdk
    ```

1. (Optional) For the unittesting, need to install E2E (End-2-End Validation) framework
    ```bash
    # install E2E package
    (mpa) .../mpa$ make -f Makefile install-e2e-package
    ```
#### __Using docker for GPU backend__

If you want to run MPA using `Docker` with `NVIDIA` GPU backend, check below things on your system.
* If your system does not have `docker`, please install it first.
  * Install Docker Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/
  * To use MPA docker image, you need to add your system user to the docker group.
    ```bash
    $ sudo usermod -aG docker $USER
    ```


* Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) on host
    ```bash
    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    ```
* Install NVIDIA Container runtime on host
    ```bash
    $ sudo apt-get install nvidia-container-runtime
    ```
* Create or update systemd drop-in file
    ```bash
    # make folder if it's not existed
    $ sudo mkdir -p /etc/systemd/system/docker.service.d
    ```
    * Create `override.conf` file at folder above with contents below
        ```
        [Service]
        ExecStart=
        ExecStart=/usr/bin/dockerd --host=fd:// --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
        ```
* Create `daemon.json` file at `/etc/docker` folder with contents below
    ```
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        },
        "insecure-registries": ["registry.toolbox.iotg.sclab.intel.com"]
    }
    ```
* Set up proxy
    * for Docker deamon
    ```bash
    $ sudo vi /etc/systemd/system/docker.service.d/http-proxy.conf
    [Service]
    Environment="HTTP_PROXY=http://proxy-dmz.intel.com:911"
    Environment="HTTPS_PROXY=http://proxy-dmz.intel.com:912"
    Environment="NO_PROXY=localhost,intel.com,192.168.0.0/16,172.16.0.0/12,127.0.0.0/8,10.0.0.0/8,/var/run/docker.sock"
    ```
    * for Docker client
    ```bash
    $ vi ~/.docker/config.json
    {
        "proxies":
        {
            "default":
            {
                "http_proxy": "http://proxy-dmz.intel.com:911/",
                "https_proxy": "http://proxy-dmz.intel.com:912/",
                "no_proxy": "localhost,intel.com,192.168.0.0/16,172.16.0.0/12,127.0.0.0/8,10.0.0.0/8,/var/run/docker.sock"
            }
        }
    }
    ```
* To apply updated configurations, you need to restart docker
    ```bash
    $ sudo systemctl daemon-reload
    $ sudo systemctl restart docker
    ```
## run-cli.sh
You can use `run-cli.sh` to run unittest or perform some task on MPA. it uses the MPA docker image which is served by internal docker registry (registry.toolbox.iotg.sclab.intel.com).
>*__notes__*: The environment to run the MPA will be configured with Docker. so you need to install `NVIDIA Container Toolkit` to use GPU. see installation instructions on [Using docker for GPU backend](#Using-docker-for-GPU-backend) section.

`run-cli.sh` will try to select proper docker image which is compatible with your host machine depending on the version of NVIDIA driver installed on the system but you can specify it through the variable `VER_CUDA`. If `$VER_CUDA` is not configured, script will get it from the result of the `nvidia-smi`.
```bash
# run CLI to get help message using docker image mpa/cu11.1/cli:latest
$ CUDA_VISIBLE_DEVICES=0 VER_CUDA=11.1 ./run-cli.sh tools.cli -h
```
Here is the list of available docker images in the registry
* registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli:latest
* registry.toolbox.iotg.sclab.intel.com/mpa/cu11.1/cli:latest


### How to run unittests
Test cases are in the `mpa/tests` folder and sub-folder `unit` contains unittest and `intg` includes some integration test cases. 

MPA is integrated with e2e test framework so you need to set some environment variable to select and run unit-tests or integration tests.
* `TT_UNIT_TESTS`: set it `True` to select all test-cases for unit-testing
* `TT_COMPONENT_TESTS`: set it `True` to select all test-cases for integration testing

```bash
# run unittests with coverage
(mpa).../mpa$ TT_UNIT_TESTS=True python -m pytest mpa tests
# run unittests with coverage using GPU backend
(mpa).../mpa$ CUDA_VISIBLE_DEVICES=0 TT_UNIT_TESTS=True python -m pytest mpa tests
# run integration tests with coverage using GPU backend
(mpa).../mpa$ CUDA_VISIBLE_DEVICES=0 TT_COMPONENT_TESTS=True python -m pytest mpa tests
```
>*__notes__*: Please refer the default `pytest` and `coverage` configurations on `mpa/pytest.ini` and `mpa/.coveragerc` respectively.

#### __Using `run-cli.sh`__
>*__notes__*: For the first attemption, it takes long time to download docker image from registry. If you meet any errors, please check it on [Open Issues](#Open_Issues).

If both `TT_UNIT_TESTS` and `TT_COMPONENT_TESTS` are not set and if the first argument of `run-cli.sh` is `pytest`, `TT_UNIT_TESTS` will be set to `True` as the default.

```bash
# run unittests using CPU backend
$ ./run-cli.sh pytest mpa tests
# run unittests using GPU backend
$ CUDA_VISIBLE_DEVICES=0 TT_UNIT_TESTS=True ./run-cli.sh pytest mpa tests
# run integration tests using GPU backend
$ CUDA_VISIBLE_DEVICES=0 TT_COMPONENT_TESTS=True ./run-cli.sh pytest mpa tests
```

To get the detailed report of the unittest coverage, you can add `--cov-report` option to the test run script. The report will be stored in `.reports/htmlcov`.
```bash
# using run-cli.sh
$ ./run-cli.sh -m pytest --cov-report html mpa tests/unit
# using local modules
(mpa).../mpa$ python -m pytest --cov-report html mpa tests/unit
```

### How to run a task on MPA
#### __Using `tools/cli.py`__
You can execute a task using command-line-interface `mpa/tools/cli.py`. Optionally, you can pass the list of GPU ids using `CUDA_VISIBLE_DEVICE` variable to use GPU for your task.
```bash
(mpa).../mpa$ python -m tools.cli -h
```
#### __Using `run-cli.sh`__
>*__notes__*: For the first attemption, it takes long time to download docker image from registry. If you meet any errors, please check it on [Open Issues](#Open_Issues).

```console
$ CUDA_VISIBLE_DEVICES=0 ./run-cli.sh tools.cli -h
```
> *__note__*: This script will create/use `logs` and `.reports` folders. The `logs` folder will be used to store result of the run CLI and the `.reports` folder will be used saving test artifact.

#### __Usage of cli__
```
usage: cli.py [-h] [--model_cfg MODEL_CFG] [--ir_path IR_PATH] [--model_ckpt MODEL_CKPT] [--data_cfg DATA_CFG] [--output_path OUTPUT_PATH] [--recipe_hparams RECIPE_HPARAMS [RECIPE_HPARAMS ...]] [--recipe_json RECIPE_JSON] [--mode {train,eval,infer,export}]
              [--stage_type {SelfSLTrainer,ClsInferrer,ClsEvaluator,ClsExporter,ClsTrainer,DetectionInferrer,DetectionEvaluator,DetectionExporter,DetectionTrainer,SegExporter,SegTrainer,HpoRunner,MdaRunner}] [--seed SEED]
              [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
              recipe_cfg

TL-Recipe Command Line Interface

positional arguments:
  recipe_cfg            Path to config file of transfer learning recipe

optional arguments:
  -h, --help            show this help message and exit
  --model_cfg MODEL_CFG
                        Path to config file of (base) model for train/infer/eval
  --ir_path IR_PATH     Path to XML file of OMZ model to load
  --model_ckpt MODEL_CKPT
                        Path to checkpoint file to load
  --data_cfg DATA_CFG   Path to config file for input data. root/file/type for each train/val/test/unlabeled dataset
  --output_path OUTPUT_PATH
                        Output directory for train or file path for infer/eval results
  --recipe_hparams RECIPE_HPARAMS [RECIPE_HPARAMS ...]
                        Override default hyper-params in the recipe, the key-value pair in xxx=yyy format will be merged into recipe config.
  --recipe_json RECIPE_JSON
                        Override/add hyper-params in the recipe, JSON configuration format will be translated as dict and merged into the recipe config.
  --mode {train,eval,infer,export}
                        Switch the run mode
  --stage_type {SelfSLTrainer,ClsInferrer,ClsEvaluator,ClsExporter,ClsTrainer,DetectionInferrer,DetectionEvaluator,DetectionExporter,DetectionTrainer,SegExporter,SegTrainer,HpoRunner,MdaRunner}
                        if stage cfg is passed through recipe_cfg argument, you can designate its type using the argument
  --seed SEED           random seed
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        set log level
```
#### __Command examples__
* How to run a task
    ```bash
    # Perform semi-supervised learning for the classification with mobilenetv2 backbone + STL10 dataset with two NVIDIA GPUs using tools/cli.py
    $ CUDA_VISIBLE_DEVICES=0 python -m tools.cli recipes/cls_da_semisl.yaml --model_cfg samples/cfgs/models/backbones/mobilenet_v2.yaml --data_cfg samples/cfgs/data/stl10_224-bs8-1k-7k.py
    # same task with above using run-cli.sh
    $ CUDA_VISIBLE_DEVICES=0 ./run-cli.sh tools.cli recipes/cls_da_semisl.yaml --model_cfg samples/cfgs/models/backbones/mobilenet_v2.yaml --data_cfg samples/cfgs/data/stl10_224-bs8-1k-7k.py
    ```
* Check output</br>
The results will be saved to `./logs`. MPA will generate sub-directory which is named as current date and time (formatted with `YYYYMMDD_HHmmSS`) and save the result to there. For the convenient retrieval, there will be the `latest` symlink in the `./logs` as well.

> *__note__*: You can utilize multiple GPUs by setting `CUDA_VISIBLE_DEVICES` variable on the GPU enabled system. for example, there are 4 GPUs and set `CUDA_VISIBLE_DEVICES=0,2`, MPA will use two GPUs the first and third for your training task. but you could not get the performance gain depending on your task configuration. it is the known issue and will be fixed later.

### How to build MPA docker image
You can bulid and push docker image for the MPA using `build.sh`.
The command below will build a docker image "registry.toolbox.iotg.sclab.intel.com/mpa/cu10.2/cli" and tag it as "1.0.1" and "latest" for the CUDA 10.2 runtime.
```bash
$ ./build.sh 1.0.1 -c 10.2
```
If you put the `-p` option to the `build.sh`, tagged images will be pushed to the docker registry.
> *__note__*: The `run-cli.sh` script will use a docker image pushed with tag `latest`. (If you set the variable "SKIP_PULL=1" before run the `run-cli.sh`, script will not pull the `latest` tag of the MPA image from registry. it will use a local image instead. it is useful to do testing and debugging your new MPA image). it means that you have to use `-p` option only when the newly built image is working correctly. And if you make change to the MPA (e.g. updated dependencies) or there are any updates on the components in the external, you have to build and push MPA images for all supported CUDA runtime version `10.2` and `11.1` using different option of `-c`.

You can get detailed information about the available options for `build.sh` using `-h` option like below.
```console
$ ./build.sh -h
USAGE: ./build.sh <tag> [Options]

Options
    -b|--build_target   Specify build target(s). choose one of ["mpa-cli", "torch-base", "all"]. default "mpa-cli"
    -p|--push           Push built image(s) to registry
    -c|--cuda           Specify CUDA version if build_target option is "torch_base" or "all"
    -t|--torch          Specify Pytorch version if build_target option is "torch_base" or "all"
    -h|--help           Print this message
```

## Open Issues
### Cannot login docker registry
Docker image for the MPA is served by internal docker registry (registry.toolbox.iotg.sclab.intel.com). but from some host machine, you cannot login into this registry. under this issue, you cannot use 'run-cli.sh' for your environment. Two options are available to address this issue.
> A. You can build docker image for the MPA by following the instructions [How to build MPA docker image](#How_to_build_MPA_docker_image) and tag it as `registry.toolbox.iotg.sclab.intel.com/mpa/${CUDA_VER}/cli:latest`

> B. Request to get the docker image for MPA as tar file.
### No performance gain from multi-GPU enabling
MPA support multi-GPUs for the training, evaluation, and inferance but some of the task does not show performance gain compared with single-GPU case. this issue will be addressed soon.
