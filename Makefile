.PHONY: install install-custom install-langchain check check-custom check-langchain test test-custom test-langchain docker-build docker-build-custom docker-build-langchain

install: install-custom install-langchain

install-custom:
	python -m pip install -r custom_version/requirements.txt

install-langchain:
	python -m pip install -r langchain_version/requirements.txt

check: check-custom check-langchain

check-custom:
	python -m py_compile custom_version/app.py custom_version/config.py custom_version/ingest.py custom_version/rag.py

check-langchain:
	python -m py_compile langchain_version/app.py langchain_version/config.py langchain_version/ingest.py langchain_version/rag.py

test: check test-custom test-langchain

test-custom:
	cd custom_version && python -m unittest discover -s tests -p "test_*.py"

test-langchain:
	cd langchain_version && python -m unittest discover -s tests -p "test_*.py"

docker-build: docker-build-custom docker-build-langchain

docker-build-custom:
	docker build -t rag-demo-custom ./custom_version

docker-build-langchain:
	docker build -t rag-demo-langchain ./langchain_version
