"""Sample-rate independence between synth instances."""

import numpy as np
import pytest

from dexed import DexedSynth, Patch


def _patch():
    patch = Patch()
    patch.algorithm = 31
    return patch


def _render(synth):
    return synth.render(
        midi_note=60, velocity=100, note_duration=0.2, render_duration=0.3
    )


def test_second_synth_at_another_rate_does_not_retune_the_first():
    """The DX7 lookup tables are process-wide, so they must be re-pointed.

    Before, constructing and loading a synth at 48 kHz rebuilt the shared
    frequency, envelope and LFO tables, and a 44.1 kHz synth that was still
    alive rendered a semitone and a half flat while still reporting 44100.
    """
    a = DexedSynth(sample_rate=44100.0)
    a.load_patch(_patch())
    before = _render(a)

    b = DexedSynth(sample_rate=48000.0)
    b.load_patch(_patch())
    _render(b)

    assert np.array_equal(_render(a), before)


def test_pitch_is_the_same_at_every_sample_rate():
    def fundamental(audio, sample_rate):
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        return np.fft.rfftfreq(len(audio), 1.0 / sample_rate)[np.argmax(spectrum)]

    for sample_rate in (22050.0, 44100.0, 48000.0):
        synth = DexedSynth(sample_rate=sample_rate)
        synth.load_patch(_patch())
        assert fundamental(_render(synth), sample_rate) == pytest.approx(262, abs=6)
