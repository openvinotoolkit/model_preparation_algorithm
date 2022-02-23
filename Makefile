PYPI_PROTOCOL?=http
PYPI_SERVER?=pypi.sclab.intel.com
PYPI_PORT?=8000
PYTHON_PIP_NO_CACHE_DIR?=
SHELL=/bin/bash -o pipefail

BUILD_NAME_SEED ?= mpu-container-id

REGISTRY ?= registry.toolbox.iotg.sclab.intel.com
MPA_IMAGE_NAME ?= mpa/mpa-cpu
MPA_IMAGE_VERSION ?= v1.1.0
OS_VERSION ?= ubuntu18.04
# MPA_IMAGE_TAG=${env.GIT_MPA_COMMIT} \
# MPA_IMAGE_NAME=${MPA_IMAGE_NAME} ${env.COMMON_PARAMETERS}
VER_MMCV ?= $(shell awk -F'==' '/mmcv-full/ {print $$2}' external/mmdetection/requirements/runtime.txt)

MPA_%:
	@ if [ "${MPA_${*}}" = "" ]; then \
		echo "Environment variable MPA_$* is not set, please set one before run"; \
		exit 1; \
	fi

prepare-build:
	cp -f .dockerignore.bak .dockerignore

build_push_image: MPA_IMAGE_TAG MPA_IMAGE_NAME
	@echo "Push MPA CPU docker image to $(REGISTRY)/$(MPA_IMAGE_NAME):$(MPA_IMAGE_TAG)"
	docker tag -t $(REGISTRY)/$(MPA_IMAGE_NAME):$(MPA_IMAGE_TAG) $(MPA_IMAGE_NAME):$(MPA_IMAGE_TAG)
	docker push $(REGISTRY)/$(MPA_IMAGE_NAME):$(MPA_IMAGE_TAG)

build_cpu_image: MPA_IMAGE_TAG MPA_IMAGE_NAME prepare-build
	echo "no CUDA version is specified."
	echo "Build docker CPU image"
	@echo "mmcv version = $(VER_MMCV)"
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
endif
	docker build $(NO_CACHE_OPTION) \
        --build-arg ver_pytorch=1.8.2 \
        --build-arg ver_mmcv=$(VER_MMCV) \
        --build-arg http_proxy=http://proxy-mu.intel.com:911 \
        --build-arg https_proxy=http://proxy-mu.intel.com:912 \
        --build-arg no_proxy=pypi.sclab.intel.com \
        -f Dockerfile.cpu.test \
        --tag $(MPA_IMAGE_NAME):$(MPA_IMAGE_TAG) .

cuda_check:
VER_CUDA ?= $(shell nvidia-smi -q | awk -F': ' '/CUDA Version/ {print $$2}')
ifeq ($(shell expr $VER_CUDA \>= 11.0), 1)
VER_CUDNN ?= 8
VER_PYTORCH ?= 1.8.2
# VER_TORCHVISION ?= 0.8.2
else
VER_CUDNN ?= 7
VER_PYTORCH ?= 1.8.2
# VER_TORCHVISION ?= 0.8.2
endif
CUDA_TAG ?= $(shell wget -q https://registry.hub.docker.com/v1/repositories/nvidia/cuda/tags -O -  | \
	sed -e 's/[][]//g' -e 's/"//g' -e 's/ //g' | tr '}' '\n'  | awk -F: '{print $$3}' | \
	grep cudnn${VER_CUDNN}-devel-${OS_VERSION} | grep ${VER_CUDA} | awk -F "" 'END{print}')

build_gpu_image: MPA_IMAGE_TAG MPA_IMAGE_NAME cuda_check prepare-build
	@echo "Build docker GPU image"
	@echo "mmcv version = $(VER_MMCV)"
ifeq ($(NO_DOCKER_CACHE),true)
	$(eval NO_CACHE_OPTION:=--no-cache)
	@echo "Docker image will be rebuilt from scratch"
endif
	docker build $(NO_CACHE_OPTION) \
	--build-arg cuda_tag=$(CUDA_TAG) \
	--build-arg ver_pytorch=$(VER_PYTORCH) \
	--build-arg ver_mmcv=$(VER_MMCV) \
	--build-arg http_proxy=http://proxy-mu.intel.com:911 \
	--build-arg https_proxy=http://proxy-mu.intel.com:912 \
	--build-arg no_proxy=pypi.sclab.intel.com \
	-f Dockerfile.gpu.test \
    --tag $(MPA_IMAGE_NAME):$(MPA_IMAGE_TAG) .


install-e2e-package: prepare-package

prepare-package:
	mkdir -p .cache\pip; \
	pip download -d .cache/pip \
				 --trusted-host $(PYPI_SERVER) \
				 --extra-index-url $(PYPI_PROTOCOL)://$(PYPI_SERVER):$(PYPI_PORT) \
				 -r ./tests/requirements.txt; \
	pip $(PYTHON_PIP_NO_CACHE_DIR) install -f .cache/pip \
				 --upgrade-strategy only-if-needed \
				 -r ./tests/requirements.txt;

ifndef BUILD_LOGS
TEST_LOG_DIR=test_log
else
TEST_LOG_DIR=$(BUILD_LOGS)
endif

ifndef TT_TESTS_DIR
TT_TESTS_DIR=$(DEFAULT_TESTS_DIR)
endif

prepare-test-dirs:
	mkdir -p $(TEST_LOG_DIR) && chmod +rw $(TEST_LOG_DIR)
ifdef TT_TEST_TEMP_DIR
	mkdir -p $(TT_TEST_TEMP_DIR) && chmod +rw $(TEST_LOG_DIR)
	export TEMP=$(TT_TEST_TEMP_DIR)
endif
	# make folder for the test reports
	mkdir -p .reports && chmod +rw .reports

tests: prepare-test-dirs MPA_IMAGE_TAG MPA_IMAGE_NAME
	@echo '============================== Printing envs ==============================='
	env || exit 0
	@echo '=========================== Printing user config ==========================='
	cat user_config.py || exit 0
	./run_tests2.sh $(TT_TESTS_DIR) | tee $(TEST_LOG_DIR)/tests.log && test $${PIPESTATUS[0]} -eq 0

cleanup_check:
ifeq ($(TT_TEST_TYPE), CPU)
PROJECT = mpa_project_cpu_name_$(BUILD_NAME_SEED)
YAML_FILES = -p $(PROJECT) -f docker-compose.cpu.test.yaml
endif
ifeq ($(TT_TEST_TYPE), GPU)
PROJECT = mpa_project_gpu_name_$(BUILD_NAME_SEED)
YAML_FILES = -p $(PROJECT) -f docker-compose.gpu.test.yaml
endif

cleanup-build:
	rm .dockerignore

cleanup: cleanup_check cleanup-build
	echo "COMMAND: docker-compose $(YAML_FILES) down"
	docker-compose $(YAML_FILES) down
