build:
	docker compose -f docker/docker-compose.yml build

connect:
	docker exec -it mario /bin/bash

down:
	docker compose -f docker/docker-compose.yml down

up:
	docker compose -f docker/docker-compose.yml up --detach

buc: build up connect