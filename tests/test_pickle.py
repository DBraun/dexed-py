#!/usr/bin/env python
"""Tests for pickling support in dexed module."""

import numpy as np
import pickle
import pytest
import dexed
from dexed import Preset


def test_pickle_basic():
    """Test that DexedSynth can be pickled and unpickled."""
    synth = dexed.DexedSynth()
    pickled = pickle.dumps(synth)
    restored = pickle.loads(pickled)
    assert restored.sample_rate == synth.sample_rate
    assert restored.sample_rate == 44100.0


def test_pickle_custom_sample_rate():
    """Test pickling with custom sample rate."""
    synth = dexed.DexedSynth(48000.0)
    restored = pickle.loads(pickle.dumps(synth))
    assert restored.sample_rate == 48000.0


def test_pickle_and_render():
    """Test that unpickled synth can render identically."""
    synth = dexed.DexedSynth(22050.0)
    preset = Preset()
    synth.load_preset(preset)
    original_audio = synth.render(midi_note=60, velocity=100,
                                  note_duration=0.1, render_duration=0.2)

    restored = pickle.loads(pickle.dumps(synth))
    restored.load_preset(preset)
    restored_audio = restored.render(midi_note=60, velocity=100,
                                     note_duration=0.1, render_duration=0.2)

    assert len(restored_audio) == len(original_audio)
    np.testing.assert_array_equal(restored_audio, original_audio)


def test_pickle_protocol_versions():
    """Test pickling with different protocol versions."""
    synth = dexed.DexedSynth(96000.0)
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(synth, protocol=protocol))
        assert restored.sample_rate == 96000.0


def test_pickle_multiple_synths():
    """Test pickling multiple synths."""
    synths = [dexed.DexedSynth(sr) for sr in [22050.0, 44100.0, 48000.0, 96000.0]]
    for original in synths:
        restored = pickle.loads(pickle.dumps(original))
        assert restored.sample_rate == original.sample_rate


def test_pickle_file_io():
    """Test pickling to/from a file."""
    import tempfile, os
    synth = dexed.DexedSynth(32000.0)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        pickle.dump(synth, f)
        temp_path = f.name

    try:
        with open(temp_path, 'rb') as f:
            restored = pickle.load(f)
        assert restored.sample_rate == 32000.0
        restored.load_preset(Preset())
        audio = restored.render(render_duration=0.1)
        assert len(audio) == int(32000 * 0.1)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
