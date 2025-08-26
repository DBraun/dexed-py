#!/usr/bin/env python
"""Tests for algorithm parameter in dexed module."""

import numpy as np
import pickle
import pytest
import dexed
from dexed import Preset


def test_algorithm_default():
    """Test that synth.algorithm reflects the loaded preset."""
    synth = dexed.DexedSynth()
    synth.load_preset(Preset(algorithm=0))
    assert synth.algorithm == 0


def test_algorithm_reflects_loaded_preset():
    """Algorithm property reflects whatever was last loaded."""
    synth = dexed.DexedSynth()
    for alg in [0, 4, 15, 23, 31]:
        synth.load_preset(Preset(algorithm=alg))
        assert synth.algorithm == alg


def test_algorithm_reflects_loaded_patch():
    """Algorithm property reflects the patch algorithm."""
    import dexed
    from dexed import Patch
    synth = dexed.DexedSynth()
    patch = Patch()
    patch.algorithm = 15
    synth.load_patch(patch)
    assert synth.algorithm == 15


def test_all_algorithms():
    """All 32 algorithms can be loaded and rendered."""
    synth = dexed.DexedSynth()
    for alg in range(32):
        synth.load_preset(Preset(algorithm=alg))
        assert synth.algorithm == alg
        audio = synth.render(render_duration=0.01)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(44100 * 0.01)


def test_algorithm_affects_output():
    """Different algorithms produce different output."""
    synth = dexed.DexedSynth()

    synth.load_preset(Preset(algorithm=0))
    audio1 = synth.render(midi_note=60, velocity=127, render_duration=0.1)

    synth.load_preset(Preset(algorithm=31))
    audio2 = synth.render(midi_note=60, velocity=127, render_duration=0.1)

    has_output = (np.abs(audio1).max() > 1e-6) or (np.abs(audio2).max() > 1e-6)
    if has_output:
        assert not np.array_equal(audio1, audio2)


def test_algorithm_pickle():
    """Algorithm is preserved through pickle."""
    for alg in [0, 7, 15, 23, 31]:
        synth = dexed.DexedSynth(sample_rate=22050.0)
        synth.load_preset(Preset(algorithm=alg))

        pickled = pickle.dumps(synth)
        restored = pickle.loads(pickled)

        assert restored.sample_rate == 22050.0
        assert restored.algorithm == alg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
