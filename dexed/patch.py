"""
DX7 Patch representation with named parameter access.

Provides high-level access to DX7 patch parameters with conversion
to/from various formats (sysex, packed, normalized arrays).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Optional
import numpy as np

# Curve names to indices
CURVE_NAMES = {
    "lin": 0, "-lin": 0, "linear": 0,
    "exp-": 1, "-exp-": 1,
    "exp+": 2, "+exp-": 2,
    "log": 3, "-log": 3,
}
CURVE_INDEX_TO_NAME = ["lin", "exp-", "exp+", "log"]

# LFO wave names to indices
LFO_WAVE_NAMES = {
    "triangle": 0, "tri": 0,
    "saw_down": 1, "saw": 1, "sawtooth": 1,
    "saw_up": 2,
    "square": 3, "sq": 3,
    "sine": 4, "sin": 4,
    "s&h": 5, "sample_hold": 5, "sh": 5,
}
LFO_WAVE_INDEX_TO_NAME = ["triangle", "saw_down", "saw_up", "square", "sine", "s&h"]


@dataclass
class Envelope:
    """4-stage DX7 envelope generator."""
    rates: List[int] = field(default_factory=lambda: [99, 99, 99, 99])
    levels: List[int] = field(default_factory=lambda: [99, 99, 99, 0])

    def __post_init__(self):
        if len(self.rates) != 4:
            raise ValueError("Envelope must have exactly 4 rates")
        if len(self.levels) != 4:
            raise ValueError("Envelope must have exactly 4 levels")


@dataclass
class PitchEnvelope:
    """Pitch envelope generator."""
    rates: List[int] = field(default_factory=lambda: [99, 99, 99, 99])
    levels: List[int] = field(default_factory=lambda: [50, 50, 50, 50])

    def __post_init__(self):
        if len(self.rates) != 4:
            raise ValueError("PitchEnvelope must have exactly 4 rates")
        if len(self.levels) != 4:
            raise ValueError("PitchEnvelope must have exactly 4 levels")


@dataclass
class LFO:
    """Low frequency oscillator settings."""
    speed: int = 35
    delay: int = 0
    pitch_mod_depth: int = 0
    amp_mod_depth: int = 0
    sync: bool = False
    _wave: int = 4  # sine

    @property
    def wave(self) -> str:
        return LFO_WAVE_INDEX_TO_NAME[self._wave]

    @wave.setter
    def wave(self, value: Union[str, int]):
        if isinstance(value, str):
            self._wave = LFO_WAVE_NAMES.get(value.lower(), 4)
        else:
            self._wave = max(0, min(5, int(value)))


@dataclass
class Operator:
    """Single DX7 operator (oscillator)."""
    # Envelope
    envelope: Envelope = field(default_factory=Envelope)

    # Output level (0-99)
    output_level: int = 99

    # Frequency settings
    frequency_coarse: int = 1  # 0-31
    frequency_fine: int = 0  # 0-99
    frequency_mode: int = 0  # 0=ratio, 1=fixed
    detune: int = 7  # 0-14, centered at 7

    # Sensitivity
    velocity_sensitivity: int = 0  # 0-7
    amp_mod_sensitivity: int = 0  # 0-3
    rate_scaling: int = 0  # 0-7

    # Keyboard level scaling
    breakpoint: int = 39  # 0-99 (corresponds to note A3)
    left_depth: int = 0  # 0-99
    right_depth: int = 0  # 0-99
    _left_curve: int = 0  # 0-3
    _right_curve: int = 0  # 0-3

    @property
    def left_curve(self) -> str:
        return CURVE_INDEX_TO_NAME[self._left_curve]

    @left_curve.setter
    def left_curve(self, value: Union[str, int]):
        if isinstance(value, str):
            self._left_curve = CURVE_NAMES.get(value.lower(), 0)
        else:
            self._left_curve = max(0, min(3, int(value)))

    @property
    def right_curve(self) -> str:
        return CURVE_INDEX_TO_NAME[self._right_curve]

    @right_curve.setter
    def right_curve(self, value: Union[str, int]):
        if isinstance(value, str):
            self._right_curve = CURVE_NAMES.get(value.lower(), 0)
        else:
            self._right_curve = max(0, min(3, int(value)))

    @property
    def frequency_ratio(self) -> float:
        """Compute frequency ratio from coarse/fine settings."""
        if self.frequency_mode == 1:  # Fixed frequency
            # Fixed frequency in Hz
            return self._compute_fixed_freq()
        else:
            # Ratio mode
            coarse = self.frequency_coarse
            if coarse == 0:
                coarse = 0.5
            return coarse + (coarse / 100.0) * self.frequency_fine

    @frequency_ratio.setter
    def frequency_ratio(self, value: float):
        """Set frequency by ratio (sets mode to ratio)."""
        self.frequency_mode = 0
        if value < 1.0:
            self.frequency_coarse = 0
            self.frequency_fine = int((value / 0.5 - 1.0) * 100)
        else:
            self.frequency_coarse = int(value)
            remainder = value - int(value)
            self.frequency_fine = int((remainder / int(value)) * 100) if int(value) > 0 else 0

    def _compute_fixed_freq(self) -> float:
        """Compute fixed frequency in Hz from coarse/fine.

        The engine computes ``logfreq = (4458616 * ((coarse & 3) * 100 + fine)) >> 3``
        (``msfa/dx7note.cc``), i.e. a decade per unit of the low two coarse bits
        and a hundredth of a decade per unit of fine. Detune is excluded, as it
        is in ratio mode.
        """
        return 10.0 ** ((self.frequency_coarse & 3) + self.frequency_fine / 100.0)


class Patch:
    """
    Complete DX7 patch with all parameters.

    Provides named access to all DX7 parameters and conversion methods
    for various formats (sysex, packed, normalized arrays).

    Operators are 0-indexed (op[0] through op[5]).
    Algorithm is 0-indexed (0-31).
    """

    def __init__(self, name: str = "INIT VOICE"):
        self.name = name[:10].ljust(10)  # DX7 names are exactly 10 chars

        # Global parameters
        self.algorithm = 0  # 0-31
        self.feedback = 0  # 0-7
        self.osc_key_sync = True
        self.transpose = 24  # 0-48, where 24 = C3

        # LFO
        self.lfo = LFO()
        self.pitch_mod_sensitivity = 0  # 0-7

        # Pitch envelope
        self.pitch_envelope = PitchEnvelope()

        # 6 operators (0-indexed)
        self._operators = [Operator() for _ in range(6)]

    @property
    def op(self) -> "_OperatorAccessor":
        """Access operators by 0-based index: patch.op[0] through patch.op[5]."""
        return _OperatorAccessor(self._operators)

    def __repr__(self) -> str:
        return f"Patch(name='{self.name.strip()}', algorithm={self.algorithm})"

    # -------------------------------------------------------------------------
    # Sysex format (156 bytes unpacked voice data)
    # -------------------------------------------------------------------------

    @classmethod
    def from_sysex(cls, data: bytes) -> "Patch":
        """
        Create Patch from 156-byte unpacked DX7 voice data.

        This is the "VCED" format used in single-voice dumps.
        """
        if len(data) < 155:
            raise ValueError(f"Sysex data must be at least 155 bytes, got {len(data)}")

        patch = cls()

        # Parse 6 operators (stored in reverse order: OP6 first)
        for op_idx in range(6):
            op = patch._operators[5 - op_idx]  # Reverse to get OP1-6 order
            base = op_idx * 21

            # Envelope rates (R1-R4)
            op.envelope.rates = [data[base + i] for i in range(4)]
            # Envelope levels (L1-L4)
            op.envelope.levels = [data[base + 4 + i] for i in range(4)]

            # Keyboard level scaling
            op.breakpoint = data[base + 8]
            op.left_depth = data[base + 9]
            op.right_depth = data[base + 10]
            op._left_curve = data[base + 11] & 0x03
            op._right_curve = data[base + 12] & 0x03

            # Other parameters
            op.rate_scaling = data[base + 13] & 0x07
            op.amp_mod_sensitivity = data[base + 14] & 0x03
            op.velocity_sensitivity = data[base + 15] & 0x07
            op.output_level = data[base + 16]
            op.frequency_mode = data[base + 17] & 0x01
            op.frequency_coarse = data[base + 18] & 0x1F
            op.frequency_fine = data[base + 19]
            op.detune = data[base + 20] & 0x0F

        # Pitch envelope
        base = 126
        patch.pitch_envelope.rates = [data[base + i] for i in range(4)]
        patch.pitch_envelope.levels = [data[base + 4 + i] for i in range(4)]

        # Global parameters
        patch.algorithm = data[134] & 0x1F
        patch.feedback = data[135] & 0x07
        patch.osc_key_sync = bool(data[136] & 0x01)

        # LFO
        patch.lfo.speed = data[137]
        patch.lfo.delay = data[138]
        patch.lfo.pitch_mod_depth = data[139]
        patch.lfo.amp_mod_depth = data[140]
        patch.lfo.sync = bool(data[141] & 0x01)
        patch.lfo._wave = data[142] & 0x07
        patch.pitch_mod_sensitivity = data[143] & 0x07

        # Transpose
        patch.transpose = data[144]

        # Name (10 ASCII characters)
        if len(data) >= 155:
            name_bytes = bytes(data[145:155])
            patch.name = name_bytes.decode("ascii", errors="replace").strip()

        return patch

    def to_sysex(self) -> bytes:
        """Export to 156-byte unpacked DX7 voice data."""
        data = bytearray(156)

        # 6 operators (stored in reverse order: OP6 first)
        for op_idx in range(6):
            op = self._operators[5 - op_idx]
            base = op_idx * 21

            # Envelope
            for i in range(4):
                data[base + i] = max(0, min(99, op.envelope.rates[i]))
                data[base + 4 + i] = max(0, min(99, op.envelope.levels[i]))

            # Keyboard level scaling
            data[base + 8] = max(0, min(99, op.breakpoint))
            data[base + 9] = max(0, min(99, op.left_depth))
            data[base + 10] = max(0, min(99, op.right_depth))
            data[base + 11] = op._left_curve & 0x03
            data[base + 12] = op._right_curve & 0x03

            # Other parameters
            data[base + 13] = op.rate_scaling & 0x07
            data[base + 14] = op.amp_mod_sensitivity & 0x03
            data[base + 15] = op.velocity_sensitivity & 0x07
            data[base + 16] = max(0, min(99, op.output_level))
            data[base + 17] = op.frequency_mode & 0x01
            data[base + 18] = op.frequency_coarse & 0x1F
            data[base + 19] = max(0, min(99, op.frequency_fine))
            data[base + 20] = op.detune & 0x0F

        # Pitch envelope
        base = 126
        for i in range(4):
            data[base + i] = max(0, min(99, self.pitch_envelope.rates[i]))
            data[base + 4 + i] = max(0, min(99, self.pitch_envelope.levels[i]))

        # Global parameters
        data[134] = self.algorithm & 0x1F
        data[135] = self.feedback & 0x07
        data[136] = 1 if self.osc_key_sync else 0

        # LFO
        data[137] = max(0, min(99, self.lfo.speed))
        data[138] = max(0, min(99, self.lfo.delay))
        data[139] = max(0, min(99, self.lfo.pitch_mod_depth))
        data[140] = max(0, min(99, self.lfo.amp_mod_depth))
        data[141] = 1 if self.lfo.sync else 0
        data[142] = self.lfo._wave & 0x07
        data[143] = self.pitch_mod_sensitivity & 0x07

        # Transpose
        data[144] = max(0, min(48, self.transpose))

        # Name (10 ASCII characters)
        name_bytes = self.name[:10].ljust(10).encode("ascii", errors="replace")
        data[145:155] = name_bytes

        # Operator on/off (all on)
        data[155] = 0x3F

        return bytes(data)

    # -------------------------------------------------------------------------
    # Packed format (128 bytes, used in bulk dumps)
    # -------------------------------------------------------------------------

    @classmethod
    def from_packed(cls, data: bytes) -> "Patch":
        """Create Patch from 128-byte packed format (used in bulk dumps)."""
        if len(data) < 128:
            raise ValueError(f"Packed data must be 128 bytes, got {len(data)}")

        # Unpack to 156-byte format
        unpacked = cls._unpack_voice(data)
        return cls.from_sysex(bytes(unpacked))

    def to_packed(self) -> bytes:
        """Export to 128-byte packed format."""
        sysex = self.to_sysex()
        return bytes(self._pack_voice(sysex))

    @staticmethod
    def _unpack_voice(packed: bytes) -> bytearray:
        """Unpack 128-byte voice to 156-byte format."""
        unpacked = bytearray(156)

        for op in range(6):
            # Copy first 11 bytes directly
            unpacked[op * 21:op * 21 + 11] = packed[op * 17:op * 17 + 11]

            # Unpack combined bytes
            left_right_curves = packed[op * 17 + 11]
            unpacked[op * 21 + 11] = left_right_curves & 0x03
            unpacked[op * 21 + 12] = (left_right_curves >> 2) & 0x03

            detune_rs = packed[op * 17 + 12]
            unpacked[op * 21 + 13] = detune_rs & 0x07
            unpacked[op * 21 + 20] = detune_rs >> 3

            kvs_ams = packed[op * 17 + 13]
            unpacked[op * 21 + 14] = kvs_ams & 0x03
            unpacked[op * 21 + 15] = kvs_ams >> 2

            unpacked[op * 21 + 16] = packed[op * 17 + 14]

            fcoarse_mode = packed[op * 17 + 15]
            unpacked[op * 21 + 17] = fcoarse_mode & 0x01
            unpacked[op * 21 + 18] = fcoarse_mode >> 1

            unpacked[op * 21 + 19] = packed[op * 17 + 16]

        # Pitch EG and other globals
        unpacked[126:135] = packed[102:111]

        oks_fb = packed[111]
        unpacked[135] = oks_fb & 0x07
        unpacked[136] = oks_fb >> 3

        unpacked[137:141] = packed[112:116]

        lpms_lfw_lks = packed[116]
        unpacked[141] = lpms_lfw_lks & 0x01
        unpacked[142] = (lpms_lfw_lks >> 1) & 0x07
        unpacked[143] = lpms_lfw_lks >> 4

        unpacked[144:155] = packed[117:128]
        unpacked[155] = 0x3F

        return unpacked

    @staticmethod
    def _pack_voice(unpacked: bytes) -> bytearray:
        """Pack 156-byte voice to 128-byte format."""
        packed = bytearray(128)

        for op in range(6):
            # Copy first 11 bytes directly
            packed[op * 17:op * 17 + 11] = unpacked[op * 21:op * 21 + 11]

            # Pack combined bytes
            packed[op * 17 + 11] = (unpacked[op * 21 + 11] & 0x03) | \
                                   ((unpacked[op * 21 + 12] & 0x03) << 2)

            packed[op * 17 + 12] = (unpacked[op * 21 + 13] & 0x07) | \
                                   ((unpacked[op * 21 + 20] & 0x0F) << 3)

            packed[op * 17 + 13] = (unpacked[op * 21 + 14] & 0x03) | \
                                   ((unpacked[op * 21 + 15] & 0x07) << 2)

            packed[op * 17 + 14] = unpacked[op * 21 + 16]

            packed[op * 17 + 15] = (unpacked[op * 21 + 17] & 0x01) | \
                                   ((unpacked[op * 21 + 18] & 0x1F) << 1)

            packed[op * 17 + 16] = unpacked[op * 21 + 19]

        # Pitch EG and other globals
        packed[102:111] = unpacked[126:135]

        packed[111] = (unpacked[135] & 0x07) | ((unpacked[136] & 0x01) << 3)

        packed[112:116] = unpacked[137:141]

        packed[116] = (unpacked[141] & 0x01) | \
                      ((unpacked[142] & 0x07) << 1) | \
                      ((unpacked[143] & 0x07) << 4)

        packed[117:128] = unpacked[144:155]

        return packed

    # -------------------------------------------------------------------------
    # Bank loading (32-voice sysex)
    # -------------------------------------------------------------------------

    @classmethod
    def load_bank(cls, filename: str) -> List["Patch"]:
        """
        Load a DX7 bank file (32 voices).

        Supports both raw 4096-byte dumps and standard sysex format.
        """
        with open(filename, "rb") as f:
            data = f.read()

        # Check for sysex header
        if data[0] == 0xF0:
            # Standard sysex format: F0 43 00 09 20 00 ... F7
            if len(data) >= 4104:  # 4096 + 8 byte header/footer
                data = data[6:6 + 4096]  # Skip header
            else:
                raise ValueError(f"Invalid sysex bank file size: {len(data)}")
        elif len(data) < 4096:
            raise ValueError(f"Bank data must be at least 4096 bytes, got {len(data)}")

        patches = []
        for i in range(32):
            packed = data[i * 128:(i + 1) * 128]
            patches.append(cls.from_packed(packed))

        return patches

    def save_to_bank(self, filename: str, patches: List["Patch"] = None):
        """Save patches to a bank file. If patches is None, saves just this patch as slot 0."""
        if patches is None:
            patches = [self] + [Patch() for _ in range(31)]

        if len(patches) != 32:
            raise ValueError("Bank must contain exactly 32 patches")

        data = bytearray()
        for patch in patches:
            data.extend(patch.to_packed())

        with open(filename, "wb") as f:
            f.write(data)

    # -------------------------------------------------------------------------
    # 155-parameter raw format (matches sysex byte layout)
    # -------------------------------------------------------------------------

    def to_raw(self) -> np.ndarray:
        """
        Export to 155-element array of raw DX7 parameter values.

        Values are in their native DX7 ranges (0-99, 0-31, etc.).
        """
        sysex = self.to_sysex()
        return np.array(list(sysex[:155]), dtype=np.uint8)

    @classmethod
    def from_raw(cls, params: np.ndarray) -> "Patch":
        """Create Patch from 155-element raw parameter array."""
        if len(params) < 155:
            raise ValueError(f"Raw params must have at least 155 elements, got {len(params)}")

        data = bytes(int(p) for p in params[:155]) + b"\x3F"
        return cls.from_sysex(data)

    def to_preset(self):
        """Convert to a :class:`~dexed.Preset` (flat, normalized ML representation)."""
        from .preset import Preset
        ops = self._operators
        return Preset(
            feedback=self.feedback / 7.0,
            transpose=self.transpose / 48.0,
            pitch_mod_sensitivity=self.pitch_mod_sensitivity / 7.0,
            lfo_speed=self.lfo.speed / 99.0,
            lfo_delay=self.lfo.delay / 99.0,
            lfo_pitch_mod_depth=self.lfo.pitch_mod_depth / 99.0,
            lfo_amp_mod_depth=self.lfo.amp_mod_depth / 99.0,
            osc_key_sync=int(self.osc_key_sync),
            lfo_sync=int(self.lfo.sync),
            pitch_env_rates=np.array(
                [r / 99.0 for r in self.pitch_envelope.rates], dtype=np.float32
            ),
            pitch_env_levels=np.array(
                [lv / 99.0 for lv in self.pitch_envelope.levels], dtype=np.float32
            ),
            op_env_rates=np.array(
                [[r / 99.0 for r in ops[i].envelope.rates] for i in range(6)],
                dtype=np.float32,
            ),
            op_env_levels=np.array(
                [[lv / 99.0 for lv in ops[i].envelope.levels] for i in range(6)],
                dtype=np.float32,
            ),
            op_output_level=np.array(
                [ops[i].output_level / 99.0 for i in range(6)], dtype=np.float32
            ),
            op_frequency_coarse=np.array(
                [ops[i].frequency_coarse / 31.0 for i in range(6)], dtype=np.float32
            ),
            op_frequency_fine=np.array(
                [ops[i].frequency_fine / 99.0 for i in range(6)], dtype=np.float32
            ),
            op_detune=np.array(
                [ops[i].detune / 14.0 for i in range(6)], dtype=np.float32
            ),
            op_velocity_sensitivity=np.array(
                [ops[i].velocity_sensitivity / 7.0 for i in range(6)], dtype=np.float32
            ),
            op_amp_mod_sensitivity=np.array(
                [ops[i].amp_mod_sensitivity / 3.0 for i in range(6)], dtype=np.float32
            ),
            op_rate_scaling=np.array(
                [ops[i].rate_scaling / 7.0 for i in range(6)], dtype=np.float32
            ),
            op_breakpoint=np.array(
                [ops[i].breakpoint / 99.0 for i in range(6)], dtype=np.float32
            ),
            op_left_depth=np.array(
                [ops[i].left_depth / 99.0 for i in range(6)], dtype=np.float32
            ),
            op_right_depth=np.array(
                [ops[i].right_depth / 99.0 for i in range(6)], dtype=np.float32
            ),
            algorithm=self.algorithm,
            lfo_wave=self.lfo._wave,
            op_frequency_mode=np.array(
                [ops[i].frequency_mode for i in range(6)], dtype=np.int32
            ),
            op_left_curve=np.array(
                [ops[i]._left_curve for i in range(6)], dtype=np.int32
            ),
            op_right_curve=np.array(
                [ops[i]._right_curve for i in range(6)], dtype=np.int32
            ),
        )



class _OperatorAccessor:
    """Helper class to provide 0-indexed operator access."""

    def __init__(self, operators: List[Operator]):
        self._operators = operators

    def __getitem__(self, idx: int) -> Operator:
        if not 0 <= idx <= 5:
            raise IndexError(f"Operator index must be 0-5, got {idx}")
        return self._operators[idx]

    def __setitem__(self, idx: int, value: Operator):
        if not 0 <= idx <= 5:
            raise IndexError(f"Operator index must be 0-5, got {idx}")
        self._operators[idx] = value

    def __iter__(self):
        return iter(self._operators)

    def __len__(self):
        return 6
