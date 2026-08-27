"""
DexedSynth wrapper with Patch and Preset support.
"""

from typing import Optional
import numpy as np

from ._dexed import DexedSynth as _DexedSynth
from .patch import Patch


class DexedSynth:
    """
    DX7 synthesizer.

    Example::

        synth = DexedSynth(sample_rate=44100)

        # From a Patch (DX7 sysex interface)
        patch = Patch.load_bank("rom1a.syx")[0]
        synth.load_patch(patch)
        audio = synth.render(midi_note=60, velocity=100)

        # From a Preset (ML interface)
        from dexed import Preset
        preset = Preset(algorithm=15, feedback=0.5)
        synth.load_preset(preset)
        audio = synth.render(midi_note=60)

    A synth may be shared between threads: loading and rendering are
    serialized per instance, and the GIL is released while audio is
    computed. For parallel rendering, give each thread its own synth so the
    work actually overlaps.
    """

    def __init__(self, sample_rate: float = 44100.0):
        self._synth = _DexedSynth(sample_rate, 0)

    def load_patch(self, patch: Patch) -> None:
        """Load a Patch into the synthesizer."""
        self._synth.load_sysex(patch.to_sysex())

    def load_preset(self, preset) -> None:
        """Load a Preset into the synthesizer."""
        self._synth.load_sysex(preset.to_patch().to_sysex())

    def render(
        self,
        midi_note: int = 60,
        velocity: int = 100,
        note_duration: float = 3.0,
        render_duration: float = 4.0,
    ) -> np.ndarray:
        """
        Render audio with current parameters.

        Args:
            midi_note: MIDI note number 0-127 (default 60 = C4)
            velocity: Note velocity 0-127 (default 100)
            note_duration: How long the note is held in seconds (default 3.0)
            render_duration: Total audio duration in seconds (default 4.0)

        Returns:
            numpy array of shape [T] with audio samples (float32)
        """
        return self._synth.render(midi_note, velocity, note_duration, render_duration)

    def render_all_ops(
        self,
        midi_note: int = 60,
        velocity: int = 100,
        note_duration: float = 3.0,
        render_duration: float = 4.0,
    ) -> np.ndarray:
        """
        Render individual operator outputs.

        Returns:
            numpy array of shape [7, T]: channels 0-5 are operators 0-5,
            channel 6 is the final output.
        """
        return self._synth.render_all_ops(midi_note, velocity, note_duration, render_duration)

    @property
    def sample_rate(self) -> float:
        """Get the sample rate in Hz."""
        return self._synth.sample_rate

    @property
    def algorithm(self) -> int:
        """Get the currently loaded DX7 algorithm (0-31)."""
        return self._synth.algorithm

    @property
    def normalize_feedback(self) -> bool:
        """
        When True, use consistent feedback scaling across all algorithms.

        When False (default), algorithms 3, 5, and 31
        have reduced feedback, matching original Dexed behavior.
        """
        return self._synth.normalize_feedback

    @normalize_feedback.setter
    def normalize_feedback(self, value: bool) -> None:
        self._synth.normalize_feedback = value

    def __getstate__(self):
        """Pickle support."""
        return self._synth.__getstate__()

    def __setstate__(self, state):
        """Pickle support."""
        self._synth = _DexedSynth.__new__(_DexedSynth)
        self._synth.__setstate__(state)
