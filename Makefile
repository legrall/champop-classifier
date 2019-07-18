#!/usr/bin/env make -f

CONTAINER_NAME = champop-classifier
current_dir := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

build_dev: 
	docker build -t $(CONTAINER_NAME):dev -f Dockerfile-dev .

dev:
	docker run -it -p 8999:5000 -v $(current_dir):/app $(CONTAINER_NAME):dev

test_dev:
	docker run -it -p 8999:5000 -v $(current_dir):/app $(CONTAINER_NAME):dev pytest

bash_dev:
	docker run -it -p 8999:5000 -v $(current_dir):/app $(CONTAINER_NAME):dev bash

build_prod:
	docker build -t $(CONTAINER_NAME): .

run:
	docker run -it -p 8998:8080 $(CONTAINER_NAME):oc
