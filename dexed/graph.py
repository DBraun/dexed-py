"""
Custom operator graph for FM synthesis with arbitrary topology.

This module provides a DX7-accurate FM synthesis implementation that allows:
- Arbitrary number of operators (not limited to 6)
- Custom modulation routing between any operators
- Per-operator feedback (including cross-operator feedback)
- Flexible carrier selection

The implementation closely follows Dexed's algorithms for:
- 4-stage envelope with accurate rate/level calculations
- Frequency calculation with coarse/fine/detune
- Output level scaling with velocity sensitivity
- Feedback calculation matching DX7 behavior

Example:
    from dexed import OperatorGraph

    # Create a 7-operator graph
    graph = OperatorGraph(num_ops=7)

    # Configure operators
    graph.op[0].output_level = 99
    graph.op[0].frequency_coarse = 1

    # Define modulation: op6 -> op5 -> op4 -> ... -> op0 (carrier)
    for i in range(5, -1, -1):
        graph.connect(i + 1, i)

    # Set carrier(s) and feedback
    graph.set_carriers([0])
    graph.set_feedback(6, 6, level=7)

    # Render
    audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                         note_duration=1.0, render_duration=1.5)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import math


# =============================================================================
# DX7 Constants and Lookup Tables (from Dexed source)
# =============================================================================

# Block size for processing (matches Dexed's N)
N = 64
LG_N = 6

# Coarse frequency multipliers (log2 scale, Q24 fixed point)
# From dx7note.cc coarsemul[]
COARSE_MUL = np.array([
    -16777216, 0, 16777216, 26591258, 33554432, 38955489, 43368474, 47099600,
    50331648, 53182516, 55732705, 58039632, 60145690, 62083076, 63876816,
    65546747, 67108864, 68576247, 69959732, 71268397, 72509921, 73690858,
    74816848, 75892776, 76922906, 77910978, 78860292, 79773775, 80654032,
    81503396, 82323963, 83117622
], dtype=np.int64)

# Level scaling lookup table (from env.cc)
LEVEL_LUT = [0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46]

# Velocity data (from dx7note.cc)
VELOCITY_DATA = np.array([
    0, 70, 86, 97, 106, 114, 121, 126, 132, 138, 142, 148, 152, 156, 160, 163,
    166, 170, 173, 174, 178, 181, 184, 186, 189, 190, 194, 196, 198, 200, 202,
    205, 206, 209, 211, 214, 216, 218, 220, 222, 224, 225, 227, 229, 230, 232,
    233, 235, 237, 238, 240, 241, 242, 243, 244, 246, 246, 248, 249, 250, 251,
    252, 253, 254
], dtype=np.int32)

# Exponential scale data for keyboard level scaling
EXP_SCALE_DATA = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 16, 19, 23, 27, 33, 39, 47, 56, 66,
    80, 94, 110, 126, 142, 158, 174, 190, 206, 222, 238, 250
], dtype=np.int32)

# Feedback bit depth
FEEDBACK_BITDEPTH = 8


def _scale_outlevel(outlevel: int) -> int:
    """Scale output level (0-99) to internal representation."""
    if outlevel >= 20:
        return 28 + outlevel
    return LEVEL_LUT[outlevel]


def _scale_velocity(velocity: int, sensitivity: int) -> int:
    """Scale velocity with sensitivity (from dx7note.cc)."""
    clamped_vel = max(0, min(127, velocity))
    vel_value = int(VELOCITY_DATA[clamped_vel >> 1]) - 239
    scaled_vel = ((sensitivity * vel_value + 7) >> 3) << 4
    return scaled_vel


def _scale_rate(midinote: int, sensitivity: int) -> int:
    """Scale envelope rate based on key (from dx7note.cc)."""
    x = min(31, max(0, midinote // 3 - 7))
    qratedelta = (sensitivity * x) >> 3
    return qratedelta


def _scale_curve(group: int, depth: int, curve: int) -> int:
    """Scale keyboard level scaling curve (from dx7note.cc)."""
    if curve == 0 or curve == 3:
        # Linear
        scale = (group * depth * 329) >> 12
    else:
        # Exponential
        raw_exp = EXP_SCALE_DATA[min(group, len(EXP_SCALE_DATA) - 1)]
        scale = (raw_exp * depth * 329) >> 15
    if curve < 2:
        scale = -scale
    return scale


def _scale_level(midinote: int, break_pt: int, left_depth: int, right_depth: int,
                 left_curve: int, right_curve: int) -> int:
    """Compute keyboard level scaling (from dx7note.cc)."""
    offset = midinote - break_pt - 17
    if offset >= 0:
        return _scale_curve((offset + 1) // 3, right_depth, right_curve)
    else:
        return _scale_curve(-(offset - 1) // 3, left_depth, left_curve)


# =============================================================================
# Sine Table (matches Dexed's Sin class)
# =============================================================================

SIN_LG_N_SAMPLES = 10
SIN_N_SAMPLES = 1 << SIN_LG_N_SAMPLES

# Pre-computed sine table (Q24 fixed point, matches Dexed)
_sintab = None


def _init_sintab():
    """Initialize sine lookup table matching Dexed's format."""
    global _sintab
    if _sintab is not None:
        return

    # Create table with delta format for interpolation
    _sintab = np.zeros(SIN_N_SAMPLES << 1, dtype=np.int32)
    for i in range(SIN_N_SAMPLES):
        y = int(round((1 << 24) * math.sin(i * 2 * math.pi / SIN_N_SAMPLES)))
        _sintab[(i << 1) + 1] = y

    # Compute deltas
    for i in range(SIN_N_SAMPLES - 1):
        _sintab[i << 1] = _sintab[(i << 1) + 3] - _sintab[(i << 1) + 1]
    _sintab[(SIN_N_SAMPLES << 1) - 2] = _sintab[1] - _sintab[(SIN_N_SAMPLES << 1) - 1]


def _sin_lookup(phase: int) -> int:
    """Lookup sine value with linear interpolation (matches Dexed)."""
    _init_sintab()
    # Ensure phase is in 24-bit range
    phase = phase & 0xffffff
    SHIFT = 24 - SIN_LG_N_SAMPLES
    lowbits = phase & ((1 << SHIFT) - 1)
    phase_int = (phase >> (SHIFT - 1)) & ((SIN_N_SAMPLES - 1) << 1)
    dy = int(_sintab[phase_int])
    y0 = int(_sintab[phase_int + 1])
    return y0 + ((dy * lowbits) >> SHIFT)


# =============================================================================
# Exp2 Table (matches Dexed's Exp2 class)
# =============================================================================

EXP2_LG_N_SAMPLES = 10
EXP2_N_SAMPLES = 1 << EXP2_LG_N_SAMPLES

_exp2tab = None


def _init_exp2tab():
    """Initialize exp2 lookup table matching Dexed's format."""
    global _exp2tab
    if _exp2tab is not None:
        return

    _exp2tab = np.zeros(EXP2_N_SAMPLES << 1, dtype=np.int64)
    inc = 2.0 ** (1.0 / EXP2_N_SAMPLES)
    y = 1 << 30
    for i in range(EXP2_N_SAMPLES):
        _exp2tab[(i << 1) + 1] = int(round(y))
        y *= inc

    for i in range(EXP2_N_SAMPLES - 1):
        _exp2tab[i << 1] = _exp2tab[(i << 1) + 3] - _exp2tab[(i << 1) + 1]
    _exp2tab[(EXP2_N_SAMPLES << 1) - 2] = (1 << 31) - _exp2tab[(EXP2_N_SAMPLES << 1) - 1]


def _exp2_lookup(x: int) -> int:
    """Lookup 2^x with linear interpolation (matches Dexed)."""
    _init_exp2tab()
    # Extract high bits for final shift, use low 24 bits for table lookup
    hibits = x >> 24
    x_low = x & 0xffffff
    SHIFT = 24 - EXP2_LG_N_SAMPLES
    lowbits = x_low & ((1 << SHIFT) - 1)
    x_int = (x_low >> (SHIFT - 1)) & ((EXP2_N_SAMPLES - 1) << 1)
    dy = int(_exp2tab[x_int])
    y0 = int(_exp2tab[x_int + 1])
    y = y0 + ((dy * lowbits) >> SHIFT)
    # Apply final shift based on integer part of x
    shift = 6 - hibits
    if shift >= 0:
        return y >> shift
    else:
        return y << (-shift)


# =============================================================================
# Frequency Lookup Table (matches Dexed's Freqlut class)
# =============================================================================

FREQ_LG_N_SAMPLES = 10
FREQ_N_SAMPLES = 1 << FREQ_LG_N_SAMPLES
MAX_LOGFREQ_INT = 20

_freqlut = None
_freqlut_sample_rate = None


def _init_freqlut(sample_rate: float):
    """Initialize frequency lookup table for given sample rate."""
    global _freqlut, _freqlut_sample_rate
    if _freqlut is not None and _freqlut_sample_rate == sample_rate:
        return

    _freqlut = np.zeros(FREQ_N_SAMPLES + 1, dtype=np.int64)
    y = (1 << (24 + MAX_LOGFREQ_INT)) / sample_rate
    inc = 2.0 ** (1.0 / FREQ_N_SAMPLES)
    for i in range(FREQ_N_SAMPLES + 1):
        _freqlut[i] = int(round(y))
        y *= inc
    _freqlut_sample_rate = sample_rate


def _freq_lookup(logfreq: int, sample_rate: float) -> int:
    """Convert log frequency to phase increment (matches Dexed)."""
    _init_freqlut(sample_rate)
    SHIFT = 24 - FREQ_LG_N_SAMPLES
    ix = (logfreq & 0xffffff) >> SHIFT
    y0 = _freqlut[ix]
    y1 = _freqlut[ix + 1]
    lowbits = logfreq & ((1 << SHIFT) - 1)
    y = y0 + (((y1 - y0) * lowbits) >> SHIFT)
    hibits = logfreq >> 24
    return int(y >> (MAX_LOGFREQ_INT - hibits))


# =============================================================================
# DX7 Envelope (matches Dexed's Env class)
# =============================================================================

class DX7Envelope:
    """
    DX7-accurate 4-stage envelope generator.

    Matches Dexed's Env class behavior including:
    - Rate-based transitions (not time-based)
    - Level scaling
    - Rising envelope jump behavior
    """

    def __init__(self, sample_rate: float = 44100.0):
        self.sample_rate = sample_rate
        self.sr_multiplier = int((44100.0 / sample_rate) * (1 << 24))
        self.initialised = False

        self.rates = [99, 99, 99, 99]
        self.levels = [99, 99, 99, 0]
        self.outlevel = 0
        self.rate_scaling = 0

        self.level = 0
        self.targetlevel = 0
        self.rising = False
        self.ix = 0
        self.down = True
        self.inc = 0

    def is_active(self) -> bool:
        """Check if envelope is active (initialised and not finished)."""
        return self.initialised and (self.ix < 4 or self.levels[3] > 0)

    def init(self, rates: List[int], levels: List[int], outlevel: int, rate_scaling: int):
        """Initialize envelope with parameters."""
        self.initialised = True
        self.rates = list(rates)
        self.levels = list(levels)
        self.outlevel = outlevel
        self.rate_scaling = rate_scaling
        self.level = 0
        self.down = True
        self._advance(0)

    def _advance(self, newix: int):
        """Advance to next envelope stage."""
        self.ix = newix
        if self.ix < 4:
            newlevel = self.levels[self.ix]
            actuallevel = _scale_outlevel(newlevel) >> 1
            actuallevel = (actuallevel << 6) + self.outlevel - 4256
            actuallevel = max(16, actuallevel)
            self.targetlevel = actuallevel << 16
            self.rising = self.targetlevel > self.level

            # Rate calculation
            qrate = (self.rates[self.ix] * 41) >> 6
            qrate += self.rate_scaling
            qrate = min(qrate, 63)

            self.inc = (4 + (qrate & 3)) << (2 + LG_N + (qrate >> 2))
            self.inc = (self.inc * self.sr_multiplier) >> 24

    def getsample(self) -> int:
        """Get current envelope level and advance."""
        if self.ix < 3 or (self.ix < 4 and not self.down):
            if self.rising:
                jumptarget = 1716
                if self.level < (jumptarget << 16):
                    self.level = jumptarget << 16
                self.level += (((17 << 24) - self.level) >> 24) * self.inc
                if self.level >= self.targetlevel:
                    self.level = self.targetlevel
                    self._advance(self.ix + 1)
            else:
                self.level -= self.inc
                if self.level <= self.targetlevel:
                    self.level = self.targetlevel
                    self._advance(self.ix + 1)

        return self.level

    def keydown(self, d: bool):
        """Handle key down/up events."""
        if self.down != d:
            self.down = d
            self._advance(0 if d else 3)


# =============================================================================
# Graph Envelope (simplified interface)
# =============================================================================

@dataclass
class GraphEnvelope:
    """4-stage envelope for graph operators."""
    rates: List[int] = field(default_factory=lambda: [99, 99, 99, 99])
    levels: List[int] = field(default_factory=lambda: [99, 99, 99, 0])

    def __post_init__(self):
        if len(self.rates) != 4:
            raise ValueError("Envelope must have exactly 4 rates")
        if len(self.levels) != 4:
            raise ValueError("Envelope must have exactly 4 levels")


# =============================================================================
# Graph Operator
# =============================================================================

@dataclass
class GraphOperator:
    """
    Operator in a custom FM synthesis graph.

    Parameters match DX7 operators for familiarity and accuracy.
    """
    output_level: int = 99          # 0-99
    frequency_coarse: int = 1       # 0-31
    frequency_fine: int = 0         # 0-99
    frequency_mode: int = 0         # 0=ratio, 1=fixed Hz
    detune: int = 7                 # 0-14 (7 = center)
    velocity_sensitivity: int = 0   # 0-7
    rate_scaling: int = 0           # 0-7
    amp_mod_sensitivity: int = 0    # 0-3

    # Keyboard level scaling
    breakpoint: int = 39            # 0-99 (C3 = 39)
    left_depth: int = 0             # 0-99
    right_depth: int = 0            # 0-99
    left_curve: int = 0             # 0=lin-, 1=exp-, 2=exp+, 3=lin+
    right_curve: int = 0            # 0=lin-, 1=exp-, 2=exp+, 3=lin+

    envelope: GraphEnvelope = field(default_factory=GraphEnvelope)

    @property
    def frequency_ratio(self) -> float:
        """Compute frequency ratio from coarse/fine settings."""
        if self.frequency_mode == 1:
            return self._fixed_frequency()
        coarse = self.frequency_coarse if self.frequency_coarse > 0 else 0.5
        fine = 1.0 + self.frequency_fine / 100.0
        return coarse * fine

    def _fixed_frequency(self) -> float:
        """Compute fixed frequency in Hz."""
        base_freqs = [1, 10, 100, 1000]
        coarse = self.frequency_coarse
        power = coarse // 4
        mult = (coarse % 4) + 1
        base = base_freqs[min(power, 3)]
        fine = 1.0 + self.frequency_fine / 100.0
        return base * mult * fine


# =============================================================================
# Operator Accessor
# =============================================================================

class _OperatorAccessor:
    """Provides 0-indexed access to operators in a graph."""

    def __init__(self, graph: 'OperatorGraph'):
        self._graph = graph

    def __getitem__(self, index: int) -> GraphOperator:
        if not 0 <= index < self._graph.num_ops:
            raise IndexError(f"Operator index must be 0-{self._graph.num_ops - 1}, got {index}")
        return self._graph._operators[index]

    def __setitem__(self, index: int, value: GraphOperator):
        if not 0 <= index < self._graph.num_ops:
            raise IndexError(f"Operator index must be 0-{self._graph.num_ops - 1}, got {index}")
        self._graph._operators[index] = value


# =============================================================================
# Operator Graph
# =============================================================================

class OperatorGraph:
    """
    Custom FM synthesis graph with arbitrary operator routing.

    Provides DX7-accurate synthesis with the flexibility to create
    custom topologies beyond the 32 standard algorithms.
    """

    def __init__(self, num_ops: int = 6):
        """
        Create a new operator graph.

        Args:
            num_ops: Number of operators (default 6, can be any positive integer)
        """
        if num_ops < 1:
            raise ValueError("Must have at least 1 operator")

        self.num_ops = num_ops
        self._operators = [GraphOperator() for _ in range(num_ops)]
        self._mod_matrix = np.zeros((num_ops, num_ops), dtype=np.float32)
        self._carriers: List[int] = []  # 0-indexed internally
        self._feedback: Dict[Tuple[int, int], int] = {}  # (source, target) -> level (0-7)
        self.op = _OperatorAccessor(self)

    def connect(self, source: int, target: int, amount: float = 1.0) -> 'OperatorGraph':
        """
        Connect source operator to modulate target operator.

        Args:
            source: Source operator index (0-indexed)
            target: Target operator index (0-indexed)
            amount: Modulation amount (default 1.0)

        Returns:
            self for method chaining
        """
        if not 0 <= source < self.num_ops:
            raise ValueError(f"Source must be 0-{self.num_ops - 1}, got {source}")
        if not 0 <= target < self.num_ops:
            raise ValueError(f"Target must be 0-{self.num_ops - 1}, got {target}")

        self._mod_matrix[target, source] = amount
        return self

    def disconnect(self, source: int, target: int) -> 'OperatorGraph':
        """Remove modulation connection between operators."""
        return self.connect(source, target, amount=0.0)

    def set_carriers(self, carriers: List[int]) -> 'OperatorGraph':
        """
        Set which operators output to audio.

        Args:
            carriers: List of operator indices (0-indexed)

        Returns:
            self for method chaining
        """
        for c in carriers:
            if not 0 <= c < self.num_ops:
                raise ValueError(f"Carrier must be 0-{self.num_ops - 1}, got {c}")
        self._carriers = list(carriers)
        return self

    def set_feedback(self, source: int, target: int, level: int = 7) -> 'OperatorGraph':
        """
        Set feedback from one operator's output to another's input.

        Args:
            source: Source operator index (0-indexed) whose output is fed back
            target: Target operator index (0-indexed) that receives feedback.
                Same as source for self-feedback.
            level: Feedback level 0-7 (0 disables, 7 is maximum)

        Returns:
            self for method chaining
        """
        if not 0 <= source < self.num_ops:
            raise ValueError(f"Source operator must be 0-{self.num_ops - 1}, got {source}")
        if not 0 <= target < self.num_ops:
            raise ValueError(f"Target operator must be 0-{self.num_ops - 1}, got {target}")
        edge = (source, target)
        if level > 0:
            self._feedback[edge] = min(7, max(0, level))
        elif edge in self._feedback:
            del self._feedback[edge]
        return self

    @property
    def mod_matrix(self) -> np.ndarray:
        """Get the modulation matrix (read-only copy)."""
        return self._mod_matrix.copy()

    @property
    def carriers(self) -> List[int]:
        """Get carrier operator indices (0-indexed)."""
        return list(self._carriers)

    @property
    def modulators(self) -> List[int]:
        """Get modulator operator indices (0-indexed)."""
        carrier_set = set(self._carriers)
        return [i for i in range(self.num_ops) if i not in carrier_set]

    # --- Query/Introspection API ---

    def get_connections(self) -> List[tuple]:
        """
        Get all connections as (source, target, amount) tuples.

        Returns:
            List of (source_op, target_op, amount) tuples, all 0-indexed.
        """
        connections = []
        for target in range(self.num_ops):
            for source in range(self.num_ops):
                amount = self._mod_matrix[target, source]
                if amount != 0:
                    connections.append((source, target, float(amount)))
        return connections

    def get_sources(self, target: int) -> List[tuple]:
        """
        Get all operators that modulate a given target.

        Args:
            target: Target operator index (0-indexed)

        Returns:
            List of (source_op, amount) tuples.
        """
        if not 0 <= target < self.num_ops:
            raise ValueError(f"Target must be 0-{self.num_ops - 1}, got {target}")
        sources = []
        for source in range(self.num_ops):
            amount = self._mod_matrix[target, source]
            if amount != 0:
                sources.append((source, float(amount)))
        return sources

    def get_targets(self, source: int) -> List[tuple]:
        """
        Get all operators that a given source modulates.

        Args:
            source: Source operator index (0-indexed)

        Returns:
            List of (target_op, amount) tuples.
        """
        if not 0 <= source < self.num_ops:
            raise ValueError(f"Source must be 0-{self.num_ops - 1}, got {source}")
        targets = []
        for target in range(self.num_ops):
            amount = self._mod_matrix[target, source]
            if amount != 0:
                targets.append((target, float(amount)))
        return targets

    def is_connected(self, source: int, target: int) -> bool:
        """Check if source modulates target."""
        if not 0 <= source < self.num_ops:
            raise ValueError(f"Source must be 0-{self.num_ops - 1}, got {source}")
        if not 0 <= target < self.num_ops:
            raise ValueError(f"Target must be 0-{self.num_ops - 1}, got {target}")
        return bool(self._mod_matrix[target, source] != 0)

    def connection_amount(self, source: int, target: int) -> float:
        """Get modulation amount (0.0 if not connected)."""
        if not 0 <= source < self.num_ops:
            raise ValueError(f"Source must be 0-{self.num_ops - 1}, got {source}")
        if not 0 <= target < self.num_ops:
            raise ValueError(f"Target must be 0-{self.num_ops - 1}, got {target}")
        return float(self._mod_matrix[target, source])

    def get_feedback(self, source: int, target: int) -> int:
        """Get feedback level for a source->target edge (0 if none)."""
        if not 0 <= source < self.num_ops:
            raise ValueError(f"Source operator must be 0-{self.num_ops - 1}, got {source}")
        if not 0 <= target < self.num_ops:
            raise ValueError(f"Target operator must be 0-{self.num_ops - 1}, got {target}")
        return self._feedback.get((source, target), 0)

    def disconnect_all(self) -> 'OperatorGraph':
        """Remove all connections (reset to isolated operators)."""
        self._mod_matrix.fill(0)
        return self

    # --- Visualization/Debug API ---

    def summary(self) -> str:
        """
        Get a human-readable summary of the graph configuration.

        Returns:
            Multi-line summary string.
        """
        lines = [f"OperatorGraph ({self.num_ops} operators)"]
        lines.append(f"  Carriers: {self.carriers}")
        lines.append(f"  Modulators: {self.modulators}")

        connections = self.get_connections()
        if connections:
            lines.append("  Connections:")
            for src, tgt, amt in connections:
                lines.append(f"    {src} -> {tgt} ({amt})")
        else:
            lines.append("  Connections: none")

        if self._feedback:
            lines.append("  Feedback:")
            for (src, tgt), level in sorted(self._feedback.items()):
                lines.append(f"    Op {src} -> Op {tgt}: level {level}")

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """
        Generate Mermaid diagram syntax for visualization.

        Returns:
            Mermaid flowchart syntax string.
        """
        lines = ["flowchart TB"]

        # Define nodes
        carrier_set = set(self._carriers)
        for i in range(self.num_ops):
            if i in carrier_set:
                lines.append(f"    op{i}[Op {i}]:::carrier")
            else:
                lines.append(f"    op{i}[Op {i}]")

        # Define connections
        for src, tgt, amt in self.get_connections():
            if amt == 1.0:
                lines.append(f"    op{src} --> op{tgt}")
            else:
                lines.append(f"    op{src} -->|{amt}| op{tgt}")

        # Define feedback loops
        for (src, tgt), level in self._feedback.items():
            lines.append(f"    op{src} -.->|fb:{level}| op{tgt}")

        # Carrier style
        lines.append("    classDef carrier fill:#90EE90")

        return "\n".join(lines)

    def to_ascii(self) -> str:
        """
        Generate ASCII art representation of the graph.

        Returns:
            Multi-line string showing operators, connections, and carriers.
        """
        lines = [f"OperatorGraph ({self.num_ops} operators)"]
        lines.append("")

        # Build adjacency info
        carrier_set = set(self._carriers)
        connections = self.get_connections()

        # Find roots (operators with no incoming modulation)
        has_incoming = set()
        for src, tgt, _ in connections:
            has_incoming.add(tgt)
        roots = [i for i in range(self.num_ops) if i not in has_incoming]

        # Simple chain representation for each root
        visited = set()
        for root in roots:
            chain = self._trace_chain(root, visited)
            if chain:
                chain_str = "  "
                for i, op in enumerate(chain):
                    if i > 0:
                        chain_str += "-->"
                    chain_str += f"[{op}]"
                    if op in carrier_set:
                        chain_str += "(C)"
                    visited.add(op)
                lines.append(chain_str)

        # Show any unvisited operators
        unvisited = [i for i in range(self.num_ops) if i not in visited]
        for op in unvisited:
            marker = "(C)" if op in carrier_set else ""
            lines.append(f"  [{op}]{marker}")

        lines.append("")
        lines.append(f"Carriers: {', '.join(map(str, self.carriers))}")

        if self._feedback:
            fb_strs = [f"{src}->{tgt} (level {level})" for (src, tgt), level in sorted(self._feedback.items())]
            lines.append(f"Feedback: {', '.join(fb_strs)}")

        return "\n".join(lines)

    def _trace_chain(self, start: int, visited: Set[int]) -> List[int]:
        """Trace a modulation chain from start operator."""
        if start in visited:
            return []
        chain = [start]
        current = start
        while True:
            targets = self.get_targets(current)
            # Follow the first unvisited target
            next_op = None
            for tgt, _ in targets:
                if tgt not in visited and tgt not in chain:
                    next_op = tgt
                    break
            if next_op is None:
                break
            chain.append(next_op)
            current = next_op
        return chain

    def _compute_processing_order(self) -> List[int]:
        """Compute operator processing order via topological sort."""
        dependencies: Dict[int, Set[int]] = {i: set() for i in range(self.num_ops)}

        for target in range(self.num_ops):
            for source in range(self.num_ops):
                if self._mod_matrix[target, source] != 0 and source != target:
                    dependencies[target].add(source)

        in_degree = {i: len(dependencies[i]) for i in range(self.num_ops)}
        queue = [i for i in range(self.num_ops) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for target in range(self.num_ops):
                if node in dependencies[target]:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)

        remaining = [i for i in range(self.num_ops) if i not in order]
        order.extend(remaining)

        return order

    def _compute_osc_freq(self, midinote: int, op: GraphOperator) -> int:
        """Compute oscillator frequency in log format (matches DX7)."""
        if op.frequency_mode == 0:
            # Ratio mode
            # Base note frequency in log scale
            logfreq = int((midinote - 69) * (1 << 24) / 12 + (69 << 24) / 12)
            # Actually, simpler: use standard 440Hz reference
            # logfreq for A4 (midi 69) = some reference value
            # The math: logfreq = log2(freq) * (1 << 24)
            # For midi note n: freq = 440 * 2^((n-69)/12)
            # So logfreq = log2(440) * (1<<24) + (n-69) * (1<<24) / 12
            log2_440 = math.log2(440.0)
            logfreq = int((log2_440 + (midinote - 69) / 12.0) * (1 << 24))

            # Apply detune (from dx7note.cc)
            detune_ratio = 0.0209 * math.exp(-0.396 * (logfreq / (1 << 24))) / 7
            logfreq += int(detune_ratio * logfreq * (op.detune - 7))

            # Apply coarse multiplier
            coarse = op.frequency_coarse & 31
            logfreq += int(COARSE_MUL[coarse])

            # Apply fine tuning
            if op.frequency_fine > 0:
                fine_offset = 24204406.323123 * math.log(1 + 0.01 * op.frequency_fine)
                logfreq += int(fine_offset)
        else:
            # Fixed frequency mode
            coarse = op.frequency_coarse
            fine = op.frequency_fine
            logfreq = (4458616 * ((coarse & 3) * 100 + fine)) >> 3
            if op.detune > 7:
                logfreq += 13457 * (op.detune - 7)

        return logfreq

    def render(
        self,
        sample_rate: float = 44100.0,
        midi_note: int = 60,
        velocity: int = 100,
        note_duration: float = 1.0,
        render_duration: float = 1.5
    ) -> np.ndarray:
        """
        Render audio from the operator graph with DX7-accurate synthesis.

        Args:
            sample_rate: Audio sample rate in Hz
            midi_note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            note_duration: How long the note is held in seconds
            render_duration: Total audio duration in seconds

        Returns:
            Audio samples as float32 numpy array
        """
        if not self._carriers:
            raise ValueError("No carriers set - use set_carriers() first")

        num_samples = int(sample_rate * render_duration)
        note_samples = int(sample_rate * note_duration)

        # Initialize envelopes for each operator
        envelopes = []
        for i, op in enumerate(self._operators):
            env = DX7Envelope(sample_rate)

            # Compute output level with all scaling
            outlevel = _scale_outlevel(op.output_level)
            level_scaling = _scale_level(
                midi_note, op.breakpoint,
                op.left_depth, op.right_depth,
                op.left_curve, op.right_curve
            )
            outlevel += level_scaling
            outlevel = min(127, outlevel)
            outlevel = outlevel << 5
            outlevel += _scale_velocity(velocity, op.velocity_sensitivity)
            outlevel = max(0, outlevel)

            rate_scaling = _scale_rate(midi_note, op.rate_scaling)
            env.init(op.envelope.rates, op.envelope.levels, outlevel, rate_scaling)
            envelopes.append(env)

        # Compute oscillator frequencies
        op_freqs = []
        for op in self._operators:
            logfreq = self._compute_osc_freq(midi_note, op)
            freq = _freq_lookup(logfreq, sample_rate)
            op_freqs.append(freq)

        # Initialize state
        phases = np.zeros(self.num_ops, dtype=np.int64)
        outputs = np.zeros(self.num_ops, dtype=np.int64)  # Q24 fixed point
        # Feedback state: keyed by (source, target) edge
        fb_bufs = {}
        fb_shifts = {}
        fb_sources: Dict[int, List[Tuple[int, int]]] = {}
        fb_targets: Dict[int, List[Tuple[int, int]]] = {}
        for (src, tgt), level in self._feedback.items():
            edge = (src, tgt)
            if level > 0:
                fb_shifts[edge] = FEEDBACK_BITDEPTH - level
            else:
                fb_shifts[edge] = 16  # Effectively disabled
            fb_bufs[edge] = [0, 0]
            fb_sources.setdefault(src, []).append(edge)
            fb_targets.setdefault(tgt, []).append(edge)

        order = self._compute_processing_order()
        output = np.zeros(num_samples, dtype=np.float64)

        # Previous block gains for interpolation (like Dexed's gain1)
        prev_gains = [0] * self.num_ops

        # Process in blocks like Dexed
        num_blocks = (num_samples + N - 1) // N
        LG_N = 6  # log2(64)

        sample_idx = 0
        for block in range(num_blocks):
            block_size = min(N, num_samples - sample_idx)

            # Check for note-off
            if sample_idx >= note_samples:
                for env in envelopes:
                    if env.down:
                        env.keydown(False)

            # Get envelope levels for this block (gain2 in Dexed)
            curr_gains = []
            dgains = []
            for op_idx, env in enumerate(envelopes):
                level = env.getsample()
                # Convert to gain using exp2
                gain2 = _exp2_lookup(level - (14 << 24))
                curr_gains.append(gain2)
                # Compute gain delta for interpolation (like Dexed's dgain)
                # dgain = (gain2 - gain1 + (N >> 1)) >> LG_N
                gain1 = prev_gains[op_idx]
                dgain = (gain2 - gain1 + (N >> 1)) >> LG_N
                dgains.append(dgain)

            # Initialize per-block gains for interpolation
            op_gains = [prev_gains[i] for i in range(self.num_ops)]

            # Process each sample in the block
            for i in range(block_size):
                # Increment gains at start of sample (like Dexed)
                for op_idx in range(self.num_ops):
                    op_gains[op_idx] += dgains[op_idx]

                # Process operators in dependency order
                for op_idx in order:
                    # Compute modulation input
                    mod_input = 0
                    for source_idx in range(self.num_ops):
                        if source_idx != op_idx:
                            weight = self._mod_matrix[op_idx, source_idx]
                            if weight != 0:
                                mod_input += int(outputs[source_idx] * weight)

                    # Target reads from feedback buffer(s)
                    for edge in fb_targets.get(op_idx, []):
                        fb_shift = fb_shifts[edge]
                        if fb_shift < 16:
                            y0 = fb_bufs[edge][0]
                            y1 = fb_bufs[edge][1]
                            scaled_fb = (y0 + y1) >> (fb_shift + 1)
                            mod_input += scaled_fb
                            fb_bufs[edge][0] = fb_bufs[edge][1]

                    # Compute operator output using sine lookup
                    phase = int(phases[op_idx])
                    y = _sin_lookup(phase + mod_input)

                    # Apply interpolated gain (use Python int to avoid overflow)
                    gain = op_gains[op_idx]
                    y = (int(y) * int(gain)) >> 24

                    # Source writes to feedback buffer(s)
                    for edge in fb_sources.get(op_idx, []):
                        fb_bufs[edge][1] = y

                    outputs[op_idx] = y

                    # Update phase
                    phases[op_idx] = (phases[op_idx] + op_freqs[op_idx]) & 0xffffffff

                # Sum carriers
                sample = 0
                for carrier_idx in self._carriers:
                    sample += outputs[carrier_idx]

                output[sample_idx + i] = sample

            # Save current gains as previous for next block
            prev_gains = curr_gains

            sample_idx += block_size

        # Convert from fixed point to float (matching C++ INT32_TO_FLOAT_SCALE = 1/(1<<25))
        output = output / (1 << 25)

        return output.astype(np.float32)

    def render_all_ops(
        self,
        sample_rate: float = 44100.0,
        midi_note: int = 60,
        velocity: int = 100,
        note_duration: float = 1.0,
        render_duration: float = 1.5
    ) -> np.ndarray:
        """
        Render individual operator outputs with DX7-accurate synthesis.

        Returns:
            Array of shape (num_ops + 1, num_samples):
            - Channels 0 to num_ops-1: Individual operator outputs
            - Last channel: Final output (sum of carriers)
        """
        if not self._carriers:
            raise ValueError("No carriers set - use set_carriers() first")

        num_samples = int(sample_rate * render_duration)
        note_samples = int(sample_rate * note_duration)

        # Initialize envelopes
        envelopes = []
        for i, op in enumerate(self._operators):
            env = DX7Envelope(sample_rate)
            outlevel = _scale_outlevel(op.output_level)
            level_scaling = _scale_level(
                midi_note, op.breakpoint,
                op.left_depth, op.right_depth,
                op.left_curve, op.right_curve
            )
            outlevel += level_scaling
            outlevel = min(127, outlevel)
            outlevel = outlevel << 5
            outlevel += _scale_velocity(velocity, op.velocity_sensitivity)
            outlevel = max(0, outlevel)
            rate_scaling = _scale_rate(midi_note, op.rate_scaling)
            env.init(op.envelope.rates, op.envelope.levels, outlevel, rate_scaling)
            envelopes.append(env)

        # Compute oscillator frequencies
        op_freqs = []
        for op in self._operators:
            logfreq = self._compute_osc_freq(midi_note, op)
            freq = _freq_lookup(logfreq, sample_rate)
            op_freqs.append(freq)

        # Initialize state
        phases = np.zeros(self.num_ops, dtype=np.int64)
        outputs = np.zeros(self.num_ops, dtype=np.int64)
        # Feedback state: keyed by (source, target) edge
        fb_bufs = {}
        fb_shifts = {}
        fb_sources: Dict[int, List[Tuple[int, int]]] = {}
        fb_targets: Dict[int, List[Tuple[int, int]]] = {}
        for (src, tgt), level in self._feedback.items():
            edge = (src, tgt)
            if level > 0:
                fb_shifts[edge] = FEEDBACK_BITDEPTH - level
            else:
                fb_shifts[edge] = 16  # Effectively disabled
            fb_bufs[edge] = [0, 0]
            fb_sources.setdefault(src, []).append(edge)
            fb_targets.setdefault(tgt, []).append(edge)

        order = self._compute_processing_order()
        all_outputs = np.zeros((self.num_ops + 1, num_samples), dtype=np.float64)

        # Previous block gains for interpolation (like Dexed's gain1)
        prev_gains = [0] * self.num_ops

        num_blocks = (num_samples + N - 1) // N
        LG_N = 6  # log2(64)
        sample_idx = 0

        for block in range(num_blocks):
            block_size = min(N, num_samples - sample_idx)

            if sample_idx >= note_samples:
                for env in envelopes:
                    if env.down:
                        env.keydown(False)

            # Get envelope levels for this block (gain2 in Dexed)
            curr_gains = []
            dgains = []
            for op_idx, env in enumerate(envelopes):
                level = env.getsample()
                gain2 = _exp2_lookup(level - (14 << 24))
                curr_gains.append(gain2)
                # Compute gain delta for interpolation
                gain1 = prev_gains[op_idx]
                dgain = (gain2 - gain1 + (N >> 1)) >> LG_N
                dgains.append(dgain)

            # Initialize per-block gains for interpolation
            op_gains = [prev_gains[i] for i in range(self.num_ops)]

            for i in range(block_size):
                # Increment gains at start of sample (like Dexed)
                for op_idx in range(self.num_ops):
                    op_gains[op_idx] += dgains[op_idx]

                for op_idx in order:
                    mod_input = 0
                    for source_idx in range(self.num_ops):
                        if source_idx != op_idx:
                            weight = self._mod_matrix[op_idx, source_idx]
                            if weight != 0:
                                mod_input += int(outputs[source_idx] * weight)

                    # Target reads from feedback buffer(s)
                    for edge in fb_targets.get(op_idx, []):
                        fb_shift = fb_shifts[edge]
                        if fb_shift < 16:
                            y0 = fb_bufs[edge][0]
                            y1 = fb_bufs[edge][1]
                            scaled_fb = (y0 + y1) >> (fb_shift + 1)
                            mod_input += scaled_fb
                            fb_bufs[edge][0] = fb_bufs[edge][1]

                    phase = int(phases[op_idx])
                    y = _sin_lookup(phase + mod_input)
                    gain = op_gains[op_idx]
                    y = (int(y) * int(gain)) >> 24

                    # Source writes to feedback buffer(s)
                    for edge in fb_sources.get(op_idx, []):
                        fb_bufs[edge][1] = y

                    outputs[op_idx] = y
                    phases[op_idx] = (phases[op_idx] + op_freqs[op_idx]) & 0xffffffff

                    # Store individual operator output
                    all_outputs[op_idx, sample_idx + i] = y

                # Sum carriers for final output
                sample = 0
                for carrier_idx in self._carriers:
                    sample += outputs[carrier_idx]
                all_outputs[self.num_ops, sample_idx + i] = sample

            # Save current gains as previous for next block
            prev_gains = curr_gains

            sample_idx += block_size

        # Convert from fixed point to float (matching C++ INT32_TO_FLOAT_SCALE = 1/(1<<25))
        all_outputs = all_outputs / (1 << 25)

        return all_outputs.astype(np.float32)

    @classmethod
    def from_matrix(
        cls,
        mod_matrix: np.ndarray,
        carriers: List[int],
        feedback: Optional[Dict[Tuple[int, int], int]] = None
    ) -> 'OperatorGraph':
        """
        Create an operator graph from a modulation matrix.

        Args:
            mod_matrix: NxN array where [i,j] = amount op j modulates op i
            carriers: List of carrier operator indices (0-indexed)
            feedback: Optional dict of {(source, target): level} for feedback (level 0-7)

        Returns:
            OperatorGraph instance
        """
        if mod_matrix.ndim != 2 or mod_matrix.shape[0] != mod_matrix.shape[1]:
            raise ValueError("mod_matrix must be square")

        num_ops = mod_matrix.shape[0]
        graph = cls(num_ops=num_ops)
        graph._mod_matrix = mod_matrix.astype(np.float32)
        graph.set_carriers(carriers)

        if feedback:
            for (src, tgt), level in feedback.items():
                graph.set_feedback(src, tgt, level)

        return graph

    @classmethod
    def from_algorithm(cls, algorithm: int) -> 'OperatorGraph':
        """
        Create an operator graph matching a DX7 algorithm.

        Args:
            algorithm: DX7 algorithm number (0-31)

        Returns:
            OperatorGraph configured to match the algorithm
        """
        from .algorithms import algorithms as alg_data

        alg = alg_data[algorithm]
        graph = cls(num_ops=6)
        graph._mod_matrix = alg.mod_matrix.astype(np.float32)
        graph.set_carriers(alg.carriers)
        # Feedback is NOT set automatically. Users should call
        # graph.set_feedback(src, tgt, level) explicitly if they want feedback.
        # The edge info is available via algorithms[n].feedback_edge

        return graph

    def __repr__(self):
        return (f"OperatorGraph(num_ops={self.num_ops}, "
                f"carriers={self.carriers}, "
                f"connections={int(np.sum(self._mod_matrix != 0))})")
