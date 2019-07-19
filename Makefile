#!/usr/bin/env make -f

CONTAINER_NAME = champop-classifier
current_dir := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

test_ci:
	true

build_dev: 
	docker build -t $(CONTAINER_NAME):dev -f Dockerfile-dev .

dev:
	docker run -it -p 8999:5000 -v $(current_dir):/app $(CONTAINER_NAME):dev

test: build_dev
	docker run -it -p 8999:5000 -v $(current_dir):/app $(CONTAINER_NAME):dev pytest

bash_dev:
	docker run -it -p 8999:5000 -v $(current_dir):/app $(CONTAINER_NAME):dev bash

image:
	docker build -t $(CONTAINER_NAME):latest .

run:
	docker run -it -p 8998:8080 $(CONTAINER_NAME):latest
