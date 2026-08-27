"""
DX7 Algorithm definitions and metadata.

Provides information about each of the 32 DX7 algorithms including:
- Which operators are carriers (output to audio)
- Which operators are modulators
- The modulation matrix showing which operators modulate which
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class Algorithm:
    """
    Information about a single DX7 algorithm.

    Instances are shared module-wide and must not be mutated: the fields are
    read-only, ``mod_matrix`` is a non-writeable array, and the accessor
    functions hand out copies of the lists.

    Attributes:
        number: Algorithm number (0-31)
        carriers: Operator indices (0-5) that output to audio
        modulators: Operator indices (0-5) that modulate other operators
        mod_matrix: read-only 6x6 matrix where mod_matrix[i,j]=1 means op j
            modulates op i
        feedback_edge: (source, target) operator indices for the feedback connection.
            For most algorithms source == target (self-feedback). Algorithms 3 and 5
            (DX7 algos 4 and 6) have cross-operator feedback.
    """
    number: int
    carriers: Tuple[int, ...]
    modulators: Tuple[int, ...]
    mod_matrix: np.ndarray
    feedback_edge: Tuple[int, int]

    def __repr__(self):
        return f"Algorithm({self.number}, carriers={self.carriers})"


# Define all 32 algorithms
# Format: (carriers, modulators, feedback_edge, mod_matrix)
# mod_matrix[i,j] = 1 means operator j modulates operator i

_ALGORITHM_DATA = {
    0: ([0, 2], [1, 3, 4, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    1: ([0, 2], [1, 3, 4, 5], (1, 1), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    2: ([0, 3], [1, 2, 4, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    3: ([0, 3], [1, 2, 4, 5], (3, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    4: ([0, 2, 4], [1, 3, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    5: ([0, 2, 4], [1, 3, 5], (4, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    6: ([0, 2], [1, 3, 4, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    7: ([0, 2], [1, 3, 4, 5], (3, 3), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    8: ([0, 2], [1, 3, 4, 5], (1, 1), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    9: ([0, 3], [1, 2, 4, 5], (2, 2), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    10: ([0, 3], [1, 2, 4, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    11: ([0, 2], [1, 3, 4, 5], (1, 1), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    12: ([0, 2], [1, 3, 4, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    13: ([0, 2], [1, 3, 4, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    14: ([0, 2], [1, 3, 4, 5], (1, 1), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    15: ([0], [1, 2, 3, 4, 5], (5, 5), [
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    16: ([0], [1, 2, 3, 4, 5], (1, 1), [
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    17: ([0], [1, 2, 3, 4, 5], (2, 2), [
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    18: ([0, 3, 4], [1, 2, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    19: ([0, 1, 3], [2, 4, 5], (2, 2), [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    20: ([0, 1, 3, 4], [2, 5], (2, 2), [
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    21: ([0, 2, 3, 4], [1, 5], (5, 5), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    22: ([0, 1, 3, 4], [2, 5], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    23: ([0, 1, 2, 3, 4], [5], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    24: ([0, 1, 2, 3, 4], [5], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    25: ([0, 1, 3], [2, 4, 5], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    26: ([0, 1, 3], [2, 4, 5], (2, 2), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    27: ([0, 2, 5], [1, 3, 4], (4, 4), [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    28: ([0, 1, 2, 4], [3, 5], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    29: ([0, 1, 2, 5], [3, 4], (4, 4), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
    30: ([0, 1, 2, 3, 4], [5], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]),
    31: ([0, 1, 2, 3, 4, 5], [], (5, 5), [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]),
}


class _AlgorithmDict:
    """Dictionary-like access to algorithms by number (0-31)."""

    def __init__(self):
        self._algorithms = {}
        for num, (carriers, modulators, fb_edge, matrix) in _ALGORITHM_DATA.items():
            mod_matrix = np.array(matrix, dtype=np.int8)
            mod_matrix.flags.writeable = False
            self._algorithms[num] = Algorithm(
                number=num,
                carriers=tuple(carriers),
                modulators=tuple(modulators),
                mod_matrix=mod_matrix,
                feedback_edge=fb_edge,
            )

    def __getitem__(self, key: int) -> Algorithm:
        if not 0 <= key <= 31:
            raise KeyError(f"Algorithm must be 0-31, got {key}")
        return self._algorithms[key]

    def __iter__(self):
        return iter(range(32))

    def __len__(self):
        return 32

    def __contains__(self, key):
        return 0 <= key <= 31


# Module-level algorithms dictionary
algorithms = _AlgorithmDict()


def get_carriers(algorithm: int) -> List[int]:
    """Get list of carrier operator indices (0-5) for an algorithm.

    Returns a fresh list; mutating it does not affect the algorithm table.
    """
    return list(algorithms[algorithm].carriers)


def get_modulators(algorithm: int) -> List[int]:
    """Get list of modulator operator indices (0-5) for an algorithm.

    Returns a fresh list; mutating it does not affect the algorithm table.
    """
    return list(algorithms[algorithm].modulators)


def get_mod_matrix(algorithm: int) -> np.ndarray:
    """Get 6x6 modulation matrix for an algorithm."""
    return algorithms[algorithm].mod_matrix.copy()


def get_feedback_edge(algorithm: int) -> Tuple[int, int]:
    """Get the feedback edge (source, target) for an algorithm."""
    return algorithms[algorithm].feedback_edge
