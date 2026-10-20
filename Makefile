.PHONY: check build
check:
	bash -n scripts/build.sh scripts/entrypoint.sh
	python3 -m unittest discover -s tests -v
build:
	bash scripts/build.sh
