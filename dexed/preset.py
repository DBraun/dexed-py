"""Preset: PyTree-compatible ML interface for DX7 parameters.

Stores all synth state including algorithm in a single object.
Continuous parameters are normalized floats in [0, 1].
Discrete parameters are plain integers or int arrays.

Registered as a JAX PyTree automatically if JAX is installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Tuple

import numpy as np

# Native DX7 max values for each continuous field.
_SCALES: Dict[str, int] = {
    "feedback": 7,
    "transpose": 48,
    "pitch_mod_sensitivity": 7,
    "lfo_speed": 99,
    "lfo_delay": 99,
    "lfo_pitch_mod_depth": 99,
    "lfo_amp_mod_depth": 99,
    "pitch_env_rates": 99,
    "pitch_env_levels": 99,
    "op_env_rates": 99,
    "op_env_levels": 99,
    "op_output_level": 99,
    "op_frequency_coarse": 31,
    "op_frequency_fine": 99,
    "op_detune": 14,
    "op_velocity_sensitivity": 7,
    "op_amp_mod_sensitivity": 3,
    "op_rate_scaling": 7,
    "op_breakpoint": 99,
    "op_left_depth": 99,
    "op_right_depth": 99,
}

# All fields are data (JAX PyTree leaves).  No meta fields — changing any
# field value never triggers JIT recompilation.
_DATA_FIELDS: List[str] = [
    # ---- Continuous floats [0, 1]  (indices 0–122) ----
    # Global (15)
    "feedback", "transpose", "pitch_mod_sensitivity",
    "lfo_speed", "lfo_delay", "lfo_pitch_mod_depth", "lfo_amp_mod_depth",
    "pitch_env_rates", "pitch_env_levels",
    # Per-operator (108)
    "op_env_rates", "op_env_levels",
    "op_output_level", "op_frequency_coarse", "op_frequency_fine",
    "op_detune", "op_velocity_sensitivity", "op_amp_mod_sensitivity",
    "op_rate_scaling", "op_breakpoint", "op_left_depth", "op_right_depth",
    # ---- Integers (indices 123–144) ----
    # Global (4): max values 1, 1, 31, 5
    "osc_key_sync", "lfo_sync", "algorithm", "lfo_wave",
    # Per-operator (18): max values 1, 3, 3 (each × 6 ops)
    "op_frequency_mode", "op_left_curve", "op_right_curve",
]

_META_FIELDS: List[str] = []

# Shape of each data field (empty tuple = scalar).
_FIELD_SHAPES: Dict[str, Tuple[int, ...]] = {
    "feedback": (), "transpose": (), "pitch_mod_sensitivity": (),
    "lfo_speed": (), "lfo_delay": (), "lfo_pitch_mod_depth": (),
    "lfo_amp_mod_depth": (),
    "osc_key_sync": (), "lfo_sync": (),
    "pitch_env_rates": (4,), "pitch_env_levels": (4,),
    "op_env_rates": (6, 4), "op_env_levels": (6, 4),
    "op_output_level": (6,), "op_frequency_coarse": (6,),
    "op_frequency_fine": (6,), "op_detune": (6,),
    "op_velocity_sensitivity": (6,), "op_amp_mod_sensitivity": (6,),
    "op_rate_scaling": (6,), "op_breakpoint": (6,),
    "op_left_depth": (6,), "op_right_depth": (6,),
    "algorithm": (), "lfo_wave": (),
    "op_frequency_mode": (6,), "op_left_curve": (6,), "op_right_curve": (6,),
}

# Fields that store integers (packed as float32, unpacked with rounding).
_INT_FIELDS = frozenset({
    "osc_key_sync", "lfo_sync",
    "algorithm", "lfo_wave",
    "op_frequency_mode", "op_left_curve", "op_right_curve",
})

_DATA_SIZE = sum(
    math.prod(s) if s else 1 for s in
    (_FIELD_SHAPES[name] for name in _DATA_FIELDS)
)  # 145

# Pre-computed leaf sizes and split indices for array_to_leaves().
_LEAF_SIZES: List[int] = [
    math.prod(_FIELD_SHAPES[n]) if _FIELD_SHAPES[n] else 1
    for n in _DATA_FIELDS
]
_LEAF_SHAPES: List[Tuple[int, ...]] = [_FIELD_SHAPES[n] for n in _DATA_FIELDS]
_SPLIT_INDICES: List[int] = []
_cumsum = 0
for _s in _LEAF_SIZES[:-1]:
    _cumsum += _s
    _SPLIT_INDICES.append(_cumsum)


@dataclass
class Preset:
    """PyTree-compatible DX7 preset with all synth state.

    All fields are JAX PyTree data leaves -- changing any value never
    triggers JIT recompilation.  This makes Preset ideal for use as the
    argument to ``jax.pure_callback``.

    ``algorithm`` is 0-indexed (0--31), matching the DX7 sysex byte.

    Because there are no meta fields, all Presets share the same treedef::

        _, treedef = jax.tree.flatten(Preset())  # works for any Preset

    Flat vector to Preset inside JIT::

        _, treedef = jax.tree.flatten(Preset())

        @jax.jit
        def make_preset(flat_params):  # flat_params is (145,)
            leaves = Preset.array_to_leaves(flat_params)
            return jax.tree.unflatten(treedef, leaves)

    Bulk serialization::

        arr = np.stack([p.to_array() for p in presets])  # (N, 145) float32
        np.save('presets.npy', arr)
        presets = [Preset.from_array(row) for row in np.load('presets.npy')]
    """

    # ---- Continuous scalars (normalized [0, 1]) ----

    feedback: float = 0.0
    transpose: float = 0.5  # 24/48, DX7 default C3
    pitch_mod_sensitivity: float = 0.0
    lfo_speed: float = 35.0 / 99.0
    lfo_delay: float = 0.0
    lfo_pitch_mod_depth: float = 0.0
    lfo_amp_mod_depth: float = 0.0

    # ---- Binary ints (0 or 1) ----

    osc_key_sync: int = 1
    lfo_sync: int = 0

    # ---- Pitch envelope (normalized [0, 1]) ----

    pitch_env_rates: np.ndarray = field(
        default_factory=lambda: np.ones(4, dtype=np.float32)
    )
    pitch_env_levels: np.ndarray = field(
        default_factory=lambda: np.full(4, 50.0 / 99.0, dtype=np.float32)
    )

    # ---- Per-operator continuous arrays (normalized [0, 1]) ----

    op_env_rates: np.ndarray = field(
        default_factory=lambda: np.ones((6, 4), dtype=np.float32)
    )
    op_env_levels: np.ndarray = field(
        default_factory=lambda: np.tile(
            np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32), (6, 1)
        )
    )
    op_output_level: np.ndarray = field(
        default_factory=lambda: np.ones(6, dtype=np.float32)
    )
    op_frequency_coarse: np.ndarray = field(
        default_factory=lambda: np.full(6, 1.0 / 31.0, dtype=np.float32)
    )
    op_frequency_fine: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    op_detune: np.ndarray = field(
        default_factory=lambda: np.full(6, 0.5, dtype=np.float32)
    )
    op_velocity_sensitivity: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    op_amp_mod_sensitivity: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    op_rate_scaling: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    op_breakpoint: np.ndarray = field(
        default_factory=lambda: np.full(6, 39.0 / 99.0, dtype=np.float32)
    )
    op_left_depth: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    op_right_depth: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )

    # ---- Categorical ints ----

    algorithm: int = 0  # 0-31
    lfo_wave: int = 4   # 0-5 (sine)
    op_frequency_mode: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )
    op_left_curve: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )
    op_right_curve: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )

    # ---- Class variables ----

    SCALES: ClassVar[Dict[str, int]] = _SCALES
    DATA_FIELDS: ClassVar[List[str]] = _DATA_FIELDS
    META_FIELDS: ClassVar[List[str]] = _META_FIELDS

    # Operator-bundle spec (for black-box per-operator use)
    GLOBAL_CONTINUOUS_SIZE: ClassVar[int] = 15
    GLOBAL_INT_MAXES: ClassVar[List[int]] = [1, 1, 31, 5]   # osc_key_sync, lfo_sync, algorithm, lfo_wave
    OP_CONTINUOUS_SIZE: ClassVar[int] = 18
    OP_INT_MAXES: ClassVar[List[int]] = [1, 3, 3]            # frequency_mode, left_curve, right_curve

    # ---- Array methods ----

    def to_array(self) -> np.ndarray:
        """Pack all parameters into a (145,) float32 array.

        Order matches ``DATA_FIELDS``.  Integer fields are cast to float32.
        """
        parts = []
        for name in _DATA_FIELDS:
            val = getattr(self, name)
            parts.append(np.asarray(val, dtype=np.float32).ravel())
        return np.concatenate(parts)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Preset:
        """Unpack a (145,) float32 array into a Preset.

        Integer fields are recovered by rounding.

        Args:
            arr: (145,) float32 array as produced by :meth:`to_array`.
        """
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if arr.size != _DATA_SIZE:
            raise ValueError(
                f"Expected {_DATA_SIZE} elements, got {arr.size}"
            )
        kwargs: dict = {}
        offset = 0
        for name in _DATA_FIELDS:
            shape = _FIELD_SHAPES[name]
            size = math.prod(shape) if shape else 1
            chunk = arr[offset : offset + size]
            if shape == ():
                if name in _INT_FIELDS:
                    kwargs[name] = int(round(float(chunk[0])))
                else:
                    kwargs[name] = float(chunk[0])
            else:
                reshaped = chunk.reshape(shape).copy()
                if name in _INT_FIELDS:
                    kwargs[name] = np.round(reshaped).astype(np.int32)
                else:
                    kwargs[name] = reshaped
            offset += size
        return cls(**kwargs)

    # ---- Operator-bundle interface (Approach B) ----

    def global_continuous(self) -> np.ndarray:
        """Global continuous params as a (15,) float32 array.

        Order: feedback, transpose, pitch_mod_sensitivity, lfo_speed,
        lfo_delay, lfo_pitch_mod_depth, lfo_amp_mod_depth,
        pitch_env_rates[0..3], pitch_env_levels[0..3].
        """
        return np.concatenate([
            np.array([self.feedback, self.transpose, self.pitch_mod_sensitivity,
                      self.lfo_speed, self.lfo_delay, self.lfo_pitch_mod_depth,
                      self.lfo_amp_mod_depth], dtype=np.float32),
            np.asarray(self.pitch_env_rates, dtype=np.float32),
            np.asarray(self.pitch_env_levels, dtype=np.float32),
        ])

    def global_ints(self) -> np.ndarray:
        """Global integer params as a (4,) int32 array.

        Order matches ``GLOBAL_INT_MAXES``:
        osc_key_sync, lfo_sync, algorithm, lfo_wave.
        """
        return np.array([self.osc_key_sync, self.lfo_sync,
                         self.algorithm, self.lfo_wave], dtype=np.int32)

    def op_continuous(self) -> np.ndarray:
        """Per-operator continuous params as a (6, 18) float32 array.

        Row ``i`` holds operator ``i``'s 18 continuous params.
        Order matches ``OP_CONTINUOUS_SIZE``:
        env_rates[4], env_levels[4], output_level, frequency_coarse,
        frequency_fine, detune, velocity_sensitivity, amp_mod_sensitivity,
        rate_scaling, breakpoint, left_depth, right_depth.
        """
        er  = np.asarray(self.op_env_rates,            dtype=np.float32)   # (6, 4)
        el  = np.asarray(self.op_env_levels,           dtype=np.float32)   # (6, 4)
        sca = np.column_stack([
            np.asarray(self.op_output_level,          dtype=np.float32),
            np.asarray(self.op_frequency_coarse,      dtype=np.float32),
            np.asarray(self.op_frequency_fine,        dtype=np.float32),
            np.asarray(self.op_detune,                dtype=np.float32),
            np.asarray(self.op_velocity_sensitivity,  dtype=np.float32),
            np.asarray(self.op_amp_mod_sensitivity,   dtype=np.float32),
            np.asarray(self.op_rate_scaling,          dtype=np.float32),
            np.asarray(self.op_breakpoint,            dtype=np.float32),
            np.asarray(self.op_left_depth,            dtype=np.float32),
            np.asarray(self.op_right_depth,           dtype=np.float32),
        ])                                                                  # (6, 10)
        return np.concatenate([er, el, sca], axis=1)                       # (6, 18)

    def op_ints(self) -> np.ndarray:
        """Per-operator integer params as a (6, 3) int32 array.

        Row ``i`` holds operator ``i``'s 3 ints.
        Order matches ``OP_INT_MAXES``:
        frequency_mode, left_curve, right_curve.
        """
        return np.column_stack([
            np.asarray(self.op_frequency_mode, dtype=np.int32),
            np.asarray(self.op_left_curve,     dtype=np.int32),
            np.asarray(self.op_right_curve,    dtype=np.int32),
        ])

    @classmethod
    def from_operator_bundles(
        cls,
        global_cont: np.ndarray,
        global_int:  np.ndarray,
        op_cont:     np.ndarray,
        op_int:      np.ndarray,
    ) -> "Preset":
        """Construct a Preset from operator-bundle arrays.

        Args:
            global_cont: (15,) float32 — global continuous params.
            global_int:  (4,)  int    — [osc_key_sync, lfo_sync, algorithm, lfo_wave].
            op_cont:     (6, 18) float32 — per-operator continuous params.
            op_int:      (6, 3)  int    — per-operator ints [freq_mode, left_curve, right_curve].

        This is the inverse of calling :meth:`global_continuous`,
        :meth:`global_ints`, :meth:`op_continuous`, :meth:`op_ints`.
        """
        gc = np.asarray(global_cont, dtype=np.float32)
        gi = np.asarray(global_int,  dtype=np.int32)
        oc = np.asarray(op_cont,     dtype=np.float32)
        oi = np.asarray(op_int,      dtype=np.int32)
        return cls(
            feedback=float(gc[0]),
            transpose=float(gc[1]),
            pitch_mod_sensitivity=float(gc[2]),
            lfo_speed=float(gc[3]),
            lfo_delay=float(gc[4]),
            lfo_pitch_mod_depth=float(gc[5]),
            lfo_amp_mod_depth=float(gc[6]),
            pitch_env_rates=gc[7:11].copy(),
            pitch_env_levels=gc[11:15].copy(),
            op_env_rates=oc[:, 0:4].copy(),
            op_env_levels=oc[:, 4:8].copy(),
            op_output_level=oc[:, 8].copy(),
            op_frequency_coarse=oc[:, 9].copy(),
            op_frequency_fine=oc[:, 10].copy(),
            op_detune=oc[:, 11].copy(),
            op_velocity_sensitivity=oc[:, 12].copy(),
            op_amp_mod_sensitivity=oc[:, 13].copy(),
            op_rate_scaling=oc[:, 14].copy(),
            op_breakpoint=oc[:, 15].copy(),
            op_left_depth=oc[:, 16].copy(),
            op_right_depth=oc[:, 17].copy(),
            osc_key_sync=int(gi[0]),
            lfo_sync=int(gi[1]),
            algorithm=int(gi[2]),
            lfo_wave=int(gi[3]),
            op_frequency_mode=oi[:, 0].copy(),
            op_left_curve=oi[:, 1].copy(),
            op_right_curve=oi[:, 2].copy(),
        )

    @staticmethod
    def array_to_leaves(arr):
        """Split a (145,) flat array into a list of PyTree leaves.

        Works with both NumPy and JAX arrays.  JIT-traceable.

        All Presets share the same treedef (no meta fields), so one
        treedef works universally::

            _, treedef = jax.tree.flatten(Preset())

            @jax.jit
            def make_preset(flat_params):
                return jax.tree.unflatten(treedef, Preset.array_to_leaves(flat_params))

        Args:
            arr: flat array with ``len(arr) == 145``.

        Returns:
            List of 28 arrays/scalars matching the PyTree leaf order.
        """
        leaves = []
        offset = 0
        for shape, size in zip(_LEAF_SHAPES, _LEAF_SIZES):
            if shape == ():
                leaves.append(arr[offset])
            else:
                leaves.append(arr[offset : offset + size].reshape(shape))
            offset += size
        return leaves

    # ---- Patch conversion ----

    @classmethod
    def from_patch(cls, patch) -> Preset:
        """Create a Preset from a :class:`~dexed.Patch`.

        Delegates to :meth:`~dexed.Patch.to_preset`.
        """
        return patch.to_preset()

    def to_patch(self, name: str = "PRESET"):
        """Convert to a :class:`~dexed.Patch`.

        Denormalizes continuous parameters, rounds to integers, and clamps
        to native DX7 ranges.
        """
        from .patch import Patch

        def _denorm(val, max_val):
            return max(0, min(max_val, int(round(float(val) * max_val))))

        patch = Patch(name=name)
        patch.algorithm = int(self.algorithm)
        patch.feedback = _denorm(self.feedback, 7)
        patch.osc_key_sync = bool(self.osc_key_sync)
        patch.transpose = _denorm(self.transpose, 48)

        patch.lfo.speed = _denorm(self.lfo_speed, 99)
        patch.lfo.delay = _denorm(self.lfo_delay, 99)
        patch.lfo.pitch_mod_depth = _denorm(self.lfo_pitch_mod_depth, 99)
        patch.lfo.amp_mod_depth = _denorm(self.lfo_amp_mod_depth, 99)
        patch.lfo.sync = bool(self.lfo_sync)
        patch.lfo._wave = int(self.lfo_wave)
        patch.pitch_mod_sensitivity = _denorm(self.pitch_mod_sensitivity, 7)

        per = np.asarray(self.pitch_env_rates)
        pel = np.asarray(self.pitch_env_levels)
        for i in range(4):
            patch.pitch_envelope.rates[i] = _denorm(float(per[i]), 99)
            patch.pitch_envelope.levels[i] = _denorm(float(pel[i]), 99)

        oer = np.asarray(self.op_env_rates)
        oel = np.asarray(self.op_env_levels)
        ofm = np.asarray(self.op_frequency_mode)
        olc = np.asarray(self.op_left_curve)
        orc = np.asarray(self.op_right_curve)
        for i in range(6):
            op = patch._operators[i]
            for j in range(4):
                op.envelope.rates[j] = _denorm(float(oer[i, j]), 99)
                op.envelope.levels[j] = _denorm(float(oel[i, j]), 99)
            op.output_level = _denorm(float(np.asarray(self.op_output_level)[i]), 99)
            op.frequency_coarse = _denorm(float(np.asarray(self.op_frequency_coarse)[i]), 31)
            op.frequency_fine = _denorm(float(np.asarray(self.op_frequency_fine)[i]), 99)
            op.detune = _denorm(float(np.asarray(self.op_detune)[i]), 14)
            op.velocity_sensitivity = _denorm(float(np.asarray(self.op_velocity_sensitivity)[i]), 7)
            op.amp_mod_sensitivity = _denorm(float(np.asarray(self.op_amp_mod_sensitivity)[i]), 3)
            op.rate_scaling = _denorm(float(np.asarray(self.op_rate_scaling)[i]), 7)
            op.breakpoint = _denorm(float(np.asarray(self.op_breakpoint)[i]), 99)
            op.left_depth = _denorm(float(np.asarray(self.op_left_depth)[i]), 99)
            op.right_depth = _denorm(float(np.asarray(self.op_right_depth)[i]), 99)
            op.frequency_mode = int(ofm[i])
            op._left_curve = int(olc[i])
            op._right_curve = int(orc[i])

        return patch


# Register as a JAX PyTree automatically if JAX is installed.
try:
    import jax as _jax
    _jax.tree_util.register_dataclass(
        Preset,
        data_fields=list(_DATA_FIELDS),
        meta_fields=[],
    )
    del _jax
except ImportError:
    pass
