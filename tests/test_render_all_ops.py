#!/usr/bin/env python
"""Tests for the render_all_ops function."""

import numpy as np
import pytest
import dexed
from dexed import Preset


def test_render_all_ops_shape():
    """Test that render_all_ops returns the correct shape."""
    synth = dexed.DexedSynth()
    synth.load_preset(Preset())

    audio = synth.render_all_ops(midi_note=60, velocity=100,
                                 note_duration=0.5, render_duration=1.0)

    assert isinstance(audio, np.ndarray)
    assert audio.shape == (7, int(44100 * 1.0))
    assert audio.dtype == np.float32


def test_render_vs_render_all_ops():
    """Verify that render_all_ops final channel matches render output."""
    patch = dexed.Patch()
    patch.algorithm = 15
    patch.feedback = 3
    patch.op[0].output_level = 99
    patch.op[0].envelope.rates = [99, 75, 50, 50]
    patch.op[0].envelope.levels = [99, 80, 60, 0]
    patch.op[1].output_level = 85
    patch.op[1].envelope.rates = [99, 70, 50, 50]
    patch.op[1].envelope.levels = [99, 70, 50, 0]

    synth = dexed.DexedSynth()
    synth.load_patch(patch)

    standard_audio = synth.render(midi_note=60, velocity=100,
                                  note_duration=0.1, render_duration=0.15)
    ops_audio = synth.render_all_ops(midi_note=60, velocity=100,
                                     note_duration=0.1, render_duration=0.15)

    final_output = ops_audio[6]
    assert len(standard_audio) == len(final_output)

    difference = np.abs(standard_audio - final_output).max()
    assert difference < 1e-3, f"Final output should match standard render (diff={difference})"


def test_different_algorithms():
    """Test render_all_ops across multiple algorithms."""
    patch = dexed.Patch()
    for i in range(6):
        patch.op[i].output_level = 90
        patch.op[i].frequency_coarse = 1
        patch.op[i].envelope.rates = [99, 70, 50, 50]
        patch.op[i].envelope.levels = [99, 80, 60, 0]

    for algo in [0, 3, 5, 15, 31]:
        patch.algorithm = algo
        synth = dexed.DexedSynth()
        synth.load_patch(patch)

        ops_audio = synth.render_all_ops(midi_note=60, velocity=100,
                                         note_duration=0.05, render_duration=0.1)

        assert ops_audio.shape == (7, int(44100 * 0.1)), f"Wrong shape for algorithm {algo}"
        assert ops_audio.dtype == np.float32


@pytest.mark.parametrize("algo", range(32))
@pytest.mark.parametrize("feedback", [0, 7])
def test_every_audible_operator_has_a_channel(algo, feedback):
    """No operator that is producing sound may come back as silence.

    DX7 algorithms 4 and 6 collapse their feedback chain into a single loop,
    and the operators inside it used to be left at zero.
    """
    patch = dexed.Patch()
    patch.algorithm = algo
    patch.feedback = feedback
    for i in range(6):
        patch.op[i].output_level = 99
        patch.op[i].envelope.rates = [99, 99, 99, 99]
        patch.op[i].envelope.levels = [99, 99, 99, 0]

    synth = dexed.DexedSynth()
    synth.load_patch(patch)
    ops_audio = synth.render_all_ops(midi_note=60, velocity=100,
                                     note_duration=0.05, render_duration=0.1)

    silent = [i for i in range(6) if not np.any(ops_audio[i])]
    assert silent == [], f"algorithm {algo}, feedback {feedback}: silent channels {silent}"


@pytest.mark.parametrize("algo", range(32))
@pytest.mark.parametrize("feedback", [0, 7])
def test_carrier_channels_sum_to_the_mix(algo, feedback):
    """The mix is the sum of the carriers the algorithm table names."""
    patch = dexed.Patch()
    patch.algorithm = algo
    patch.feedback = feedback
    for i in range(6):
        patch.op[i].output_level = 99
        patch.op[i].envelope.rates = [99, 99, 99, 99]
        patch.op[i].envelope.levels = [99, 99, 99, 0]

    synth = dexed.DexedSynth()
    synth.load_patch(patch)
    ops_audio = synth.render_all_ops(midi_note=60, velocity=100,
                                     note_duration=0.05, render_duration=0.1)

    carriers = dexed.get_carriers(algo)
    error = np.abs(ops_audio[carriers].sum(axis=0) - ops_audio[6]).max()
    assert error < 1e-6, f"algorithm {algo}, feedback {feedback}: mix error {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
