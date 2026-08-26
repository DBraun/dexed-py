"""Python bindings for Dexed DX7 synthesizer"""

from .version import __version__

try:
    from ._dexed import DexedSynth as _RawDexedSynth
except ImportError as e:
    import sys
    raise ImportError(
        f"Failed to import _dexed C++ extension: {e}\n"
        "Make sure the module is properly built and installed."
    ) from e

from .patch import Patch
from .preset import Preset
from .synth import DexedSynth
from .algorithms import (
    algorithms,
    get_carriers,
    get_modulators,
    get_mod_matrix,
    get_feedback_edge,
)
from .graph import OperatorGraph, GraphOperator, GraphEnvelope

__all__ = [
    "DexedSynth",
    "Patch",
    "Preset",
    "OperatorGraph",
    "GraphOperator",
    "GraphEnvelope",
    "algorithms",
    "get_carriers",
    "get_modulators",
    "get_mod_matrix",
    "get_feedback_edge",
]
