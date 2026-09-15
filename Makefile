.PHONY: install install-full test lint format api gradio docker-build docker-up clean

install:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[dev,depth,sfm,masking,splat,onnx]"

test:
	pytest tests/ -v --cov=depthcraft

lint:
	ruff check depthcraft/
	mypy depthcraft/ --ignore-missing-imports

format:
	ruff check depthcraft/ --fix

api:
	uvicorn depthcraft.api.main:app --host 0.0.0.0 --port 8000 --reload

gradio:
	python -m depthcraft.visualization.gradio_app

mvp:
	python mvp_demo.py --image $(IMAGE) --headless

docker-build:
	docker build -t depthcraft:latest -f docker/Dockerfile.cpu .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

clean:
	rm -rf outputs/* __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
