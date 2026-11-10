.PHONY: test test-slow db-up db-down track embed embed-all

PYTHON ?= python
PYTEST ?= pytest
DOCKER_COMPOSE ?= docker compose

test:
	$(PYTEST) -q

test-slow:
	$(PYTEST) -q -m slow

db-up:
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --env-file docker/.env up -d

db-down:
	$(DOCKER_COMPOSE) -f docker/docker-compose.yml --env-file docker/.env down

track:
	$(PYTHON) -m src.vision.tracker --video "$(VIDEO)" --video-id "$(ID)" --config configs/vision.yaml

env:
	@echo "Use conda activate video_rag first, then run: pip install -r requirements.txt"

embed:
	$(PYTHON) -m src.vision.embed_crops --video-id "$(ID)" --config configs/vision.yaml --no-db

enbed-all:
	$(PYTHON) -m src.vision.embed_crops --all --config configs/vision.yaml --no-db
