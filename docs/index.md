# dexed-py

> Python bindings for the [Dexed](https://github.com/asb2m10/dexed) DX7 synthesizer, with support for high-level patch editing, ML/JAX workflows, and low-level parameter arrays.

**Source**: [github.com/DBraun/dexed-py](https://github.com/DBraun/dexed-py) | **PyPI**: [dexed-py](https://pypi.org/project/dexed-py/)

An [llms.txt](llms.txt) file is available for LLM context.

## Features

- **High-level Patch API** for creating and editing DX7 patches with named parameters
- **DX7 sysex support** for loading and saving `.syx` bank files and individual voice dumps
- **Preset — ML/JAX interface** with 145-element normalized flat arrays, full JAX PyTree support, and zero JIT recompilation overhead
- **Custom operator graphs** that go beyond the 32 fixed DX7 algorithms (arbitrary topologies, any number of operators)
- **Per-operator rendering** for analyzing individual operator contributions
- **Algorithm metadata** including carrier/modulator classification and modulation matrices

## Requirements

- Python >= 3.11
- NumPy
- Optional: [JAX](https://github.com/jax-ml/jax) for PyTree integration

## Installation

```bash
pip install dexed-py
```

```{toctree}
:maxdepth: 2

quickstart
api
parameter-format
changelog
```
