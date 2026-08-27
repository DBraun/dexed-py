# Simple developer Makefile for dexed-py
# The actual build is handled by scikit-build-core

.PHONY: all clean install test wheel develop docs chart

all: install

clean:
	rm -rf build/ dist/ *.egg-info dexed_py.egg-info/
	rm -f dexed/*.so dexed/*.pyd dexed/_*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

install: clean
	pip install .

develop:
	pip install -e .

wheel: clean
	python -m build --wheel

test:
	python -m pytest tests -v

docs:
	cd docs && make html

# Redraw docs/dx7_algorithms.svg from the vendored routing table.
# Add PNG=docs/dx7_algorithms.png to also rasterize (needs cairosvg).
chart:
	python docs/make_algorithm_chart.py $(if $(PNG),--png $(PNG),)

# Install build dependencies
deps:
	pip install --upgrade pip build
	pip install scikit-build-core cmake