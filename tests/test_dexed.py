#!/usr/bin/env python
"""Tests for dexed module."""

import numpy as np
import pytest
import dexed
from dexed import Preset


def test_import():
    """Test that the module can be imported."""
    assert hasattr(dexed, 'DexedSynth')


def test_synth_creation():
    """Test creating a synth with default and custom sample rates."""
    synth1 = dexed.DexedSynth()
    assert synth1.sample_rate == 44100.0

    synth2 = dexed.DexedSynth(48000.0)
    assert synth2.sample_rate == 48000.0


def test_render_basic():
    """Test basic rendering functionality."""
    synth = dexed.DexedSynth()
    synth.load_preset(Preset())
    audio = synth.render()

    assert isinstance(audio, np.ndarray)
    assert audio.shape == (int(44100 * 4.0),)
    assert audio.dtype == np.float32


def test_render_with_parameters():
    """Test rendering with different MIDI parameters."""
    synth = dexed.DexedSynth()
    synth.load_preset(Preset())

    audio1 = synth.render(midi_note=60)
    audio2 = synth.render(midi_note=72)
    assert len(audio1) == len(audio2)

    audio3 = synth.render(velocity=64)
    audio4 = synth.render(velocity=127)
    assert len(audio3) == len(audio4)


def test_render_durations():
    """Test rendering with different duration parameters."""
    synth = dexed.DexedSynth()
    synth.load_preset(Preset())

    audio1 = synth.render(render_duration=0.1)
    audio2 = synth.render(render_duration=0.5)
    assert len(audio1) == int(44100 * 0.1)
    assert len(audio2) == int(44100 * 0.5)

    audio3 = synth.render(note_duration=0.1, render_duration=0.2)
    assert len(audio3) == int(44100 * 0.2)


def test_render_without_loading():
    """Test that rendering without loading params raises an error."""
    synth = dexed.DexedSynth()
    with pytest.raises(RuntimeError, match="Parameters must be loaded"):
        synth.render()


def test_render_multiple_notes():
    """Test that parameters can be loaded once and used for multiple renders."""
    synth = dexed.DexedSynth()
    synth.load_preset(Preset())

    audio1 = synth.render(midi_note=60)
    audio2 = synth.render(midi_note=64)
    audio3 = synth.render(midi_note=67)

    assert all(len(a) == int(44100 * 4.0) for a in [audio1, audio2, audio3])


@pytest.mark.parametrize("byte134", [8, 40, 72, 200, 255])
def test_load_sysex_bounds_the_algorithm_byte(byte134):
    """Byte 134 indexes a 32-entry table in Dx7Note::init, unmasked.

    load_sysex memcpy'd all 156 bytes verbatim, so an out-of-range byte read
    past the end of FmCore::algorithms; .algorithm reported a masked value that
    did not match what was rendered.
    """
    from dexed._dexed import DexedSynth as RawSynth

    patch = dexed.Patch()
    for i in range(6):
        patch.op[i].output_level = 99
    data = bytearray(patch.to_sysex())
    data[134] = byte134

    synth = RawSynth(44100.0, 0)
    synth.load_sysex(bytes(data))
    assert synth.algorithm == byte134 & 0x1F

    reference = RawSynth(44100.0, 0)
    data[134] = byte134 & 0x1F
    reference.load_sysex(bytes(data))
    assert np.array_equal(
        synth.render(60, 100, 0.05, 0.1), reference.render(60, 100, 0.05, 0.1)
    )


@pytest.mark.parametrize("byte135", [7, 9, 10, 100, 255])
def test_load_sysex_bounds_the_feedback_byte(byte135):
    """Feedback >= 10 made the kernels shift by a negative count."""
    from dexed._dexed import DexedSynth as RawSynth

    patch = dexed.Patch()
    patch.algorithm = 31
    for i in range(6):
        patch.op[i].output_level = 99
    data = bytearray(patch.to_sysex())
    data[135] = byte135

    synth = RawSynth(44100.0, 0)
    synth.load_sysex(bytes(data))

    reference = RawSynth(44100.0, 0)
    data[135] = byte135 & 0x07
    reference.load_sysex(bytes(data))
    assert np.array_equal(
        synth.render(60, 100, 0.05, 0.1), reference.render(60, 100, 0.05, 0.1)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
