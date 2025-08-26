#!/usr/bin/env python
"""Tests for JAX PyTree integration.  Skipped if JAX is not installed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from dexed import DexedSynth, Preset


def test_tree_flatten_unflatten():
    """Flatten and unflatten a Preset through JAX tree_util."""
    p = Preset(algorithm=5, feedback=0.3, lfo_wave=2)
    leaves, treedef = jax.tree_util.tree_flatten(p)

    # All 28 fields are leaves
    assert len(leaves) == len(Preset.DATA_FIELDS)

    p2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert p2.algorithm == 5
    assert p2.lfo_wave == 2
    np.testing.assert_allclose(float(p2.feedback), 0.3, atol=1e-6)


def test_single_treedef():
    """All Presets share the same treedef — no meta fields."""
    _, td1 = jax.tree.flatten(Preset(algorithm=0))
    _, td2 = jax.tree.flatten(Preset(algorithm=31, lfo_wave=3))
    assert td1 == td2


def test_no_recompile_on_any_field_change():
    """Changing any field — including algorithm — must NOT cause recompilation."""
    @jax.jit
    def identity(preset):
        return jax.tree.leaves(preset)

    identity(Preset(algorithm=0, osc_key_sync=1, lfo_sync=0))
    assert identity._cache_size() == 1

    identity(Preset(algorithm=5))
    assert identity._cache_size() == 1, "algorithm change should NOT recompile"

    identity(Preset(algorithm=5, lfo_wave=3))
    assert identity._cache_size() == 1, "lfo_wave change should NOT recompile"

    identity(Preset(algorithm=5, osc_key_sync=0, lfo_sync=1))
    assert identity._cache_size() == 1, "osc_key_sync/lfo_sync change should NOT recompile"


def test_pure_callback_render():
    """Render audio inside a JAX computation via pure_callback."""
    synth = DexedSynth()

    def render_cb(preset: Preset) -> jnp.ndarray:
        synth.load_preset(preset)
        return synth.render(midi_note=60, velocity=100, note_duration=0.5, render_duration=1.0)

    @jax.jit
    def jitted_render(preset: Preset) -> jnp.ndarray:
        return jax.pure_callback(
            render_cb, jax.ShapeDtypeStruct((44100,), jnp.float32), preset,
        )

    audio = jitted_render(Preset(
        algorithm=0, feedback=0.5, op_output_level=jnp.full(6, 0.8),
    ))
    assert audio.shape == (44100,)
    assert jnp.max(jnp.abs(audio)) > 0


def test_array_to_leaves_jit_traceable():
    """array_to_leaves + tree.unflatten works inside JIT, single compilation."""
    synth = DexedSynth()
    _, treedef = jax.tree.flatten(Preset())  # universal treedef

    def render_cb(preset):
        synth.load_preset(preset)
        return synth.render(midi_note=60, velocity=100, note_duration=0.5, render_duration=1.0)

    @jax.jit
    def render_from_flat(flat_params):
        preset = jax.tree.unflatten(treedef, Preset.array_to_leaves(flat_params))
        return jax.pure_callback(
            render_cb, jax.ShapeDtypeStruct((44100,), jnp.float32), preset,
        )

    flat = jnp.array(Preset(algorithm=0, feedback=0.5).to_array())
    audio = render_from_flat(flat)
    assert audio.shape == (44100,)
    assert jnp.max(jnp.abs(audio)) > 0

    # Different values — same algorithm — must not recompile
    flat2 = jnp.array(Preset(algorithm=0, feedback=0.8).to_array())
    render_from_flat(flat2)

    # Different algorithm embedded in flat array — still no recompile
    flat3 = jnp.array(Preset(algorithm=15, feedback=0.4).to_array())
    render_from_flat(flat3)

    assert render_from_flat._cache_size() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
