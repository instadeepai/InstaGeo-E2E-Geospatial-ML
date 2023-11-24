#################
# General setup #
#################

GIT_REPO=instadeepai/ml-research-template
GIT_BRANCH=main
CHECKOUT_DIR=ml-research-template

# The following variables are assumed to already exist as environment variables locally.
# Alternatively, the below can be uncommented and edited - however don't push access tokens
# to external repositories!

#GITLAB_USER_TOKEN=$(GITLAB_USER_TOKEN)
#GITLAB_ACCESS_TOKEN=$(GITLAB_ACCESS_TOKEN)

#######
# TPU #
#######
# Shared set-up.
BASE_CMD=gcloud alpha compute tpus tpu-vm
BASE_NAME=ml-research-template-tpu
WORKER=all
PORT=8889
NUM_DEVICES=8

# Basic TPU configuration.
# For details on available TPU's see the Notion page:
#    https://www.notion.so/instadeep/TPU-Cloud-Computing-for-Research-aea6a9cf28ec44d481ed49dd54e191be
PROJECT=research-294715
ZONE=us-central1-f
ACCELERATOR_TYPE=v2-8
NAME=$(BASE_NAME)-$(ACCELERATOR_TYPE)
RUNTIME_VERSION=v2-alpha

.PHONY: set_project
set_project:
	gcloud config set project $(PROJECT)

.PHONY: create_vm
create_vm:
	$(BASE_CMD) create $(NAME) --zone $(ZONE) \
		--project $(PROJECT) \
		--accelerator-type $(ACCELERATOR_TYPE) \
		--version $(RUNTIME_VERSION) \

.PHONY: prepare_vm
prepare_vm:
	$(BASE_CMD) ssh --zone $(ZONE) $(NAME) \
		--project $(PROJECT) \
		--worker=$(WORKER) \
		--command="git clone -b ${GIT_BRANCH} https://${GITHUB_USER_TOKEN}:${GITHUB_ACCESS_TOKEN}@github.com/${GIT_REPO}.git ${CHECKOUT_DIR}"

.PHONY: create
create: create_vm prepare_vm

.PHONY: start
start:
	$(BASE_CMD) start $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: connect
connect:
	$(BASE_CMD) ssh $(NAME) --zone $(ZONE) --project $(PROJECT)

.PHONY: list
list:
	$(BASE_CMD) list --zone=$(ZONE) --project $(PROJECT)

.PHONY: describe
describe:
	$(BASE_CMD) describe $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: stop
stop:
	$(BASE_CMD) stop $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: delete
delete:
	$(BASE_CMD) delete $(NAME) --zone $(ZONE) --project $(PROJECT)

.PHONY: run
run:
	$(BASE_CMD) ssh --zone $(ZONE) $(NAME) --project $(PROJECT) --worker=$(WORKER) --command="$(command)"

##########
# Docker #
##########

SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_ARGS = \
	--build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID) \
	--build-arg GITLAB_USER_TOKEN=$(GITLAB_USER_TOKEN) \
	--build-arg GITLAB_ACCESS_TOKEN=$(GITLAB_ACCESS_TOKEN)

DOCKER_RUN_FLAGS = --rm --privileged -p ${PORT}:${PORT} --network host
DOCKER_VARS_TO_PASS =
DOCKER_IMAGE_NAME = research_template
DOCKER_CONTAINER_NAME = research_template_container


.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rfv
	find . | grep -E ".pytest_cache" | xargs rm -rfv
	find . | grep -E "nul" | xargs rm -rfv

.PHONY: docker_build
docker_build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) --build-arg ACCELERATOR=$(ACCELERATOR) -f docker/Dockerfile .

.PHONY: docker_build_tpu
docker_build_tpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) --build-arg ACCELERATOR="tpu" -f docker/Dockerfile .

.PHONY: docker_build_cpu
docker_build_cpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) --build-arg ACCELERATOR="cpu" -f docker/Dockerfile .

.PHONY: docker_build_cuda
docker_build_cuda:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) --build-arg ACCELERATOR="cuda" -f docker/Dockerfile .

.PHONY: docker_run
docker_run:
	sudo docker run $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) $(command)

.PHONY: docker_start
docker_start:
	sudo docker run -itd $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME)

.PHONY: docker_enter
docker_enter:
	sudo docker exec -it $(DOCKER_CONTAINER_NAME) /bin/bash

.PHONY: docker_kill
docker_kill:
	sudo docker kill $(DOCKER_CONTAINER_NAME)

.PHONY: docker_list
docker_list:
	sudo docker ps
