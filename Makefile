
NPROCS                     := $(shell nproc)
VERSION                    := $(shell grep -oP '(?<=__version__ = ")[^"]*' vllm/version.py)
DOCKER_TEST_IMAGE_VLLM     := harbor.h2o.ai/h2ogpt/test-image-vllm:$(VERSION)

ifeq ($(VERSION),)
  $(error Failed to extract version number from vllm/version.py)
endif

VLLM_CUDA_VERSION ?= 12.1.0
VLLM_BASE_IMAGE   ?= gcr.io/vorvan/h2oai/h2ogpt-vllm-wolfi-base:1

docker_build:
	docker pull $(VLLM_BASE_IMAGE)
	docker pull nvidia/cuda:$(VLLM_CUDA_VERSION)-devel-ubuntu22.04
	docker buildx build --load --build-arg max_jobs=$(NPROCS) --build-arg PYTHON_VERSION=3.10 --build-arg CUDA_VERSION=$(VLLM_CUDA_VERSION) --build-arg WOLFI_OS_BASE_IMAGE=$(VLLM_BASE_IMAGE) --tag $(DOCKER_TEST_IMAGE_VLLM) --file Dockerfile .

docker_push:
	docker tag $(DOCKER_TEST_IMAGE_VLLM) gcr.io/vorvan/h2oai/h2ogpte-vllm:$(VERSION)
