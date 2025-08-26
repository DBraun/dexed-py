#!/usr/bin/env python
"""Tests for the Preset class."""

import numpy as np
import pytest

from dexed import DexedSynth, Patch, Preset


class TestPresetDefaults:
    def test_default_matches_init_voice(self):
        default = Preset()
        from_patch = Preset.from_patch(Patch())
        assert default.algorithm == from_patch.algorithm
        assert default.osc_key_sync == from_patch.osc_key_sync
        assert default.lfo_sync == from_patch.lfo_sync
        assert default.lfo_wave == from_patch.lfo_wave
        np.testing.assert_array_equal(default.op_frequency_mode, from_patch.op_frequency_mode)
        np.testing.assert_array_equal(default.op_left_curve, from_patch.op_left_curve)
        np.testing.assert_array_equal(default.op_right_curve, from_patch.op_right_curve)
        assert abs(default.feedback - from_patch.feedback) < 1e-6
        np.testing.assert_allclose(default.to_array(), from_patch.to_array(), atol=1e-6)

    def test_field_counts(self):
        assert len(Preset.DATA_FIELDS) == 28
        assert len(Preset.META_FIELDS) == 0

    def test_algorithm_0_indexed(self):
        assert Preset().algorithm == 0
        patch = Patch()
        patch.algorithm = 0
        assert Preset.from_patch(patch).algorithm == 0
        patch.algorithm = 31
        assert Preset.from_patch(patch).algorithm == 31

    def test_osc_key_sync_is_int(self):
        p = Preset()
        assert isinstance(p.osc_key_sync, int)
        assert p.osc_key_sync == 1

    def test_lfo_sync_is_int(self):
        p = Preset()
        assert isinstance(p.lfo_sync, int)
        assert p.lfo_sync == 0

    def test_op_frequency_mode_is_array(self):
        p = Preset()
        assert isinstance(p.op_frequency_mode, np.ndarray)
        assert p.op_frequency_mode.shape == (6,)
        np.testing.assert_array_equal(p.op_frequency_mode, [0, 0, 0, 0, 0, 0])


class TestToArray:
    def test_shape_and_dtype(self):
        arr = Preset().to_array()
        assert arr.shape == (145,)
        assert arr.dtype == np.float32

    def test_continuous_values_in_range(self):
        arr = Preset().to_array()
        # Continuous fields are in [0, 1]; int fields are small non-negative ints
        assert arr.min() >= 0.0

    def test_custom_values(self):
        p = Preset(feedback=1.0, op_output_level=np.zeros(6, dtype=np.float32))
        arr = p.to_array()
        assert arr[0] == 1.0  # feedback is first

class TestFromArray:
    def test_round_trip(self):
        p = Preset(
            algorithm=15,
            feedback=0.7,
            lfo_wave=2,
            osc_key_sync=0,
            lfo_sync=1,
            op_output_level=np.full(6, 0.8, dtype=np.float32),
            op_frequency_mode=np.array([1, 0, 1, 0, 0, 0], dtype=np.int32),
        )
        p2 = Preset.from_array(p.to_array())
        np.testing.assert_allclose(p2.to_array(), p.to_array(), atol=1e-6)
        assert p2.algorithm == 15
        assert p2.lfo_wave == 2
        assert p2.osc_key_sync == 0
        assert p2.lfo_sync == 1
        np.testing.assert_array_equal(p2.op_frequency_mode, [1, 0, 1, 0, 0, 0])

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="145"):
            Preset.from_array(np.zeros(100))

class TestBulkSerialization:
    def test_stack_and_restore(self):
        presets = [Preset(algorithm=i, feedback=i / 31.0) for i in range(32)]
        arr = np.stack([p.to_array() for p in presets])
        assert arr.shape == (32, 145)
        restored = [Preset.from_array(row) for row in arr]
        for orig, rest in zip(presets, restored):
            assert rest.algorithm == orig.algorithm
            assert abs(rest.feedback - orig.feedback) < 1e-6

    def test_numpy_save_load(self, tmp_path):
        presets = [Preset(algorithm=i) for i in range(10)]
        arr = np.stack([p.to_array() for p in presets])
        path = tmp_path / "presets.npy"
        np.save(path, arr)
        loaded = np.load(path)
        restored = [Preset.from_array(row) for row in loaded]
        for orig, rest in zip(presets, restored):
            assert rest.algorithm == orig.algorithm


class TestPatchConversion:
    def test_from_patch_round_trip(self):
        patch = Patch()
        patch.algorithm = 15
        patch.feedback = 5
        patch.osc_key_sync = False
        patch.lfo.wave = "square"
        patch.op[0].output_level = 80
        patch.op[2].frequency_coarse = 4
        patch.op[2].left_curve = "exp+"
        patch.op[2].right_curve = "log"

        preset = Preset.from_patch(patch)
        assert preset.algorithm == 15
        assert preset.osc_key_sync == 0

        restored = preset.to_patch()
        assert restored.algorithm == 15
        assert restored.feedback == 5
        assert restored.osc_key_sync is False
        assert restored.lfo.wave == "square"
        assert restored.op[0].output_level == 80
        assert restored.op[2].frequency_coarse == 4
        assert restored.op[2].left_curve == "exp+"
        assert restored.op[2].right_curve == "log"

    def test_preset_round_trip_close(self):
        p = Preset(
            algorithm=7,
            feedback=0.42857142857,
            op_output_level=np.full(6, 0.75, dtype=np.float32),
        )
        p2 = Preset.from_patch(p.to_patch())
        assert p2.algorithm == 7
        assert abs(p2.feedback - p.feedback) < 0.08
        np.testing.assert_allclose(p2.op_output_level, p.op_output_level, atol=1.0 / 99 + 1e-6)

    def test_all_operators_preserved(self):
        patch = Patch()
        for i in range(6):
            patch.op[i].output_level = 10 * (i + 1)
            patch.op[i].frequency_coarse = i + 1
            patch.op[i].detune = i + 2

        preset = Preset.from_patch(patch)
        restored = preset.to_patch()
        for i in range(6):
            assert restored.op[i].output_level == patch.op[i].output_level
            assert restored.op[i].frequency_coarse == patch.op[i].frequency_coarse
            assert restored.op[i].detune == patch.op[i].detune

    def test_op_curves_round_trip(self):
        patch = Patch()
        for i in range(6):
            patch.op[i]._left_curve = (i + 1) % 4
            patch.op[i]._right_curve = (i + 2) % 4

        preset = Preset.from_patch(patch)
        assert preset.op_left_curve.dtype == np.int32
        restored = preset.to_patch()
        for i in range(6):
            assert restored.op[i]._left_curve == patch.op[i]._left_curve
            assert restored.op[i]._right_curve == patch.op[i]._right_curve


class TestAudioEquivalence:
    def test_same_audio(self):
        patch = Patch()
        patch.algorithm = 0
        patch.feedback = 4
        patch.op[0].output_level = 99
        patch.op[0].envelope.rates = [99, 99, 99, 99]
        patch.op[0].envelope.levels = [99, 99, 99, 0]

        preset = Preset.from_patch(patch)
        synth = DexedSynth()

        synth.load_patch(patch)
        audio_patch = synth.render(midi_note=60, velocity=100, note_duration=0.2, render_duration=0.3)

        synth.load_preset(preset)
        audio_preset = synth.render(midi_note=60, velocity=100, note_duration=0.2, render_duration=0.3)

        np.testing.assert_allclose(audio_preset, audio_patch, atol=1e-6)

    def test_complex_patch_same_audio(self):
        patch = Patch()
        patch.algorithm = 15
        patch.feedback = 7
        patch.lfo.wave = "saw_down"
        for i in range(6):
            patch.op[i].output_level = 80
            patch.op[i].frequency_coarse = i + 1
            patch.op[i].envelope.rates = [99, 80, 50, 30]
            patch.op[i].envelope.levels = [99, 80, 60, 0]

        preset = Preset.from_patch(patch)
        synth = DexedSynth()

        synth.load_patch(patch)
        audio_patch = synth.render(midi_note=64, velocity=100, note_duration=0.2, render_duration=0.3)

        synth.load_preset(preset)
        audio_preset = synth.render(midi_note=64, velocity=100, note_duration=0.2, render_duration=0.3)

        np.testing.assert_allclose(audio_preset, audio_patch, atol=1e-6)


class TestLoadPreset:
    def test_algorithm_set(self):
        synth = DexedSynth()
        synth.load_preset(Preset(algorithm=23))
        assert synth.algorithm == 23

    def test_produces_audio(self):
        synth = DexedSynth()
        synth.load_preset(Preset(
            algorithm=0,
            feedback=0.5,
            op_output_level=np.full(6, 0.8, dtype=np.float32),
        ))
        audio = synth.render(midi_note=60, velocity=100, note_duration=0.5, render_duration=1.0)
        assert audio.shape == (44100,)
        assert np.max(np.abs(audio)) > 0

class TestArrayToLeaves:
    def test_leaf_count(self):
        leaves = Preset.array_to_leaves(Preset().to_array())
        assert len(leaves) == len(Preset.DATA_FIELDS)  # 28

    def test_round_trip_via_leaves(self):
        p = Preset(
            algorithm=5,
            feedback=0.6,
            osc_key_sync=0,
            lfo_sync=1,
            op_output_level=np.full(6, 0.42, dtype=np.float32),
        )
        arr = p.to_array()
        leaves = Preset.array_to_leaves(arr)
        assert abs(float(leaves[0]) - 0.6) < 1e-6       # feedback  (index 0)
        assert float(leaves[21]) == 0.0               # osc_key_sync (index 21)
        assert float(leaves[22]) == 1.0               # lfo_sync     (index 22)
        np.testing.assert_allclose(leaves[11], 0.42, atol=1e-6)  # op_output_level (index 11)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestOperatorBundles:
    """Tests for the per-operator bundle interface (Approach B)."""

    def test_shapes(self):
        p = Preset()
        assert p.global_continuous().shape == (Preset.GLOBAL_CONTINUOUS_SIZE,)
        assert p.global_ints().shape == (len(Preset.GLOBAL_INT_MAXES),)
        assert p.op_continuous().shape == (6, Preset.OP_CONTINUOUS_SIZE)
        assert p.op_ints().shape == (6, len(Preset.OP_INT_MAXES))

    def test_dtypes(self):
        p = Preset()
        assert p.global_continuous().dtype == np.float32
        assert p.global_ints().dtype == np.int32
        assert p.op_continuous().dtype == np.float32
        assert p.op_ints().dtype == np.int32

    def test_global_continuous_content(self):
        p = Preset(feedback=0.5, lfo_speed=0.3)
        gc = p.global_continuous()
        assert abs(gc[0] - 0.5) < 1e-6   # feedback
        assert abs(gc[3] - 0.3) < 1e-6   # lfo_speed

    def test_global_ints_content(self):
        p = Preset(algorithm=15, lfo_wave=3, osc_key_sync=0, lfo_sync=1)
        gi = p.global_ints()
        assert gi[0] == 0   # osc_key_sync
        assert gi[1] == 1   # lfo_sync
        assert gi[2] == 15  # algorithm
        assert gi[3] == 3   # lfo_wave

    def test_op_continuous_rows(self):
        levels = np.linspace(0.1, 0.6, 6, dtype=np.float32)
        p = Preset(op_output_level=levels)
        oc = p.op_continuous()
        np.testing.assert_allclose(oc[:, 8], levels, atol=1e-6)  # output_level at col 8

    def test_op_ints_content(self):
        lc = np.array([0, 1, 2, 3, 0, 1], dtype=np.int32)
        p = Preset(op_left_curve=lc)
        oi = p.op_ints()
        np.testing.assert_array_equal(oi[:, 1], lc)  # left_curve at col 1

    def test_round_trip(self):
        patch = Patch()
        patch.algorithm = 15
        patch.feedback = 5
        patch.lfo.wave = "square"
        for i in range(6):
            patch.op[i].output_level = 10 * (i + 1)
            patch.op[i].frequency_coarse = i + 1
            patch.op[i]._left_curve = (i + 1) % 4

        p = patch.to_preset()
        p2 = Preset.from_operator_bundles(
            p.global_continuous(), p.global_ints(),
            p.op_continuous(), p.op_ints(),
        )
        np.testing.assert_allclose(p2.to_array(), p.to_array(), atol=1e-6)

    def test_maxes_respected(self):
        p = Preset()
        gc = p.global_continuous()
        gi = p.global_ints()
        oc = p.op_continuous()
        oi = p.op_ints()
        assert gc.min() >= 0.0 and gc.max() <= 1.0
        for val, mx in zip(gi, Preset.GLOBAL_INT_MAXES):
            assert 0 <= val <= mx
        assert oc.min() >= 0.0 and oc.max() <= 1.0
        for col, mx in enumerate(Preset.OP_INT_MAXES):
            assert oi[:, col].min() >= 0 and oi[:, col].max() <= mx
