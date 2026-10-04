CONTAINER_RUNTIME ?= podman
IMAGES := $(shell $(CONTAINER_RUNTIME) images -f "dangling=true" -q)
CONTAINERS := $(shell $(CONTAINER_RUNTIME) ps -a -q -f status=exited)
VERSION := 0.1
REPOSITORY := messydesk
IMAGE := md-gensim
LOCAL_IMAGE := localhost/$(REPOSITORY)/$(IMAGE):$(VERSION)
SHORT_IMAGE := $(REPOSITORY)/$(IMAGE):$(VERSION)


clean:
	-$(CONTAINER_RUNTIME) rm -f $(CONTAINERS)
	-$(CONTAINER_RUNTIME) rmi -f $(IMAGES)

build:
	$(CONTAINER_RUNTIME) build -t $(LOCAL_IMAGE) .
	$(CONTAINER_RUNTIME) tag $(LOCAL_IMAGE) $(SHORT_IMAGE)


start:
	$(CONTAINER_RUNTIME) run --rm -it --name $(IMAGE) -p 9009:9009 $(LOCAL_IMAGE)


restart:
	-$(CONTAINER_RUNTIME) stop $(IMAGE)
	-$(CONTAINER_RUNTIME) rm $(IMAGE)
	$(MAKE) start

bash:
	$(CONTAINER_RUNTIME) exec -it $(IMAGE) bash

# Unit tests in a throwaway container (gensim has no wheels for every Python).
test: build
	$(CONTAINER_RUNTIME) run --rm -v $(CURDIR)/tests:/app/tests:ro,Z $(LOCAL_IMAGE) \
		sh -c "pip install -q pytest && python -m pytest -q tests"
