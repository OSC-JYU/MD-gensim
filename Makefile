IMAGES := $(shell docker images -f "dangling=true" -q)
CONTAINERS := $(shell docker ps -a -q -f status=exited)
VOLUME := md-gensim
VERSION := 0.1
REPOSITORY := local
IMAGE := md-gensim


clean:
	docker rm -f $(CONTAINERS)
	docker rmi -f $(IMAGES)

build:
	docker build -t $(REPOSITORY)/messydesk/$(IMAGE):$(VERSION) .

start:
	docker run -d --name $(IMAGE) \
		-p 9009:9009 \
		-e MD_URL=http://host.containers.internal:8200 \
		--restart unless-stopped \
		$(REPOSITORY)/messydesk/$(IMAGE):$(VERSION)
stop:
	docker stop $(IMAGE)
	docker rm $(IMAGE)
	
restart:
	$(MAKE) stop
	$(MAKE) start

bash:
	docker exec -it $(IMAGE) bash

