#!/usr/bin/env python
"""Tests for the Patch class and related functionality."""

import numpy as np
import pytest
from dexed import (
    Patch, DexedSynth, algorithms, get_carriers, get_modulators, get_mod_matrix
)
from dexed.patch import Operator, Envelope, LFO


class TestPatchCreation:
    """Tests for Patch creation and basic properties."""

    def test_create_default_patch(self):
        """Test creating a patch with default values."""
        patch = Patch()
        assert patch.algorithm == 0
        assert patch.feedback == 0
        assert patch.osc_key_sync is True
        assert patch.transpose == 24
        assert len(patch.name) == 10

    def test_create_named_patch(self):
        """Test creating a patch with a name."""
        patch = Patch(name="Bass")
        assert patch.name.strip() == "Bass"

    def test_name_truncation(self):
        """Test that long names are truncated to 10 characters."""
        patch = Patch(name="VeryLongPatchName")
        assert len(patch.name) == 10
        assert patch.name == "VeryLongPa"

    def test_operator_access(self):
        """Test 0-indexed operator access."""
        patch = Patch()

        # Access operators 0-5
        for i in range(6):
            op = patch.op[i]
            assert isinstance(op, Operator)

        # Invalid indices should raise
        with pytest.raises(IndexError):
            _ = patch.op[-1]
        with pytest.raises(IndexError):
            _ = patch.op[6]

    def test_operator_modification(self):
        """Test modifying operator parameters."""
        patch = Patch()

        patch.op[0].output_level = 99
        patch.op[0].frequency_coarse = 2
        patch.op[0].frequency_fine = 50

        assert patch.op[0].output_level == 99
        assert patch.op[0].frequency_coarse == 2
        assert patch.op[0].frequency_fine == 50


class TestEnvelope:
    """Tests for Envelope class."""

    def test_envelope_defaults(self):
        """Test envelope default values."""
        env = Envelope()
        assert len(env.rates) == 4
        assert len(env.levels) == 4

    def test_envelope_custom(self):
        """Test envelope with custom values."""
        env = Envelope(rates=[99, 80, 50, 30], levels=[99, 90, 70, 0])
        assert env.rates == [99, 80, 50, 30]
        assert env.levels == [99, 90, 70, 0]

    def test_envelope_invalid_length(self):
        """Test that invalid envelope lengths raise errors."""
        with pytest.raises(ValueError):
            Envelope(rates=[99, 80, 50])  # Only 3 rates

        with pytest.raises(ValueError):
            Envelope(levels=[99, 90])  # Only 2 levels


class TestLFO:
    """Tests for LFO class."""

    def test_lfo_defaults(self):
        """Test LFO default values."""
        lfo = LFO()
        assert lfo.speed == 35
        assert lfo.delay == 0
        assert lfo.wave == "sine"

    def test_lfo_wave_by_name(self):
        """Test setting LFO wave by name."""
        lfo = LFO()
        lfo.wave = "triangle"
        assert lfo.wave == "triangle"

        lfo.wave = "square"
        assert lfo.wave == "square"

    def test_lfo_wave_by_index(self):
        """Test setting LFO wave by index."""
        lfo = LFO()
        lfo.wave = 0
        assert lfo.wave == "triangle"

        lfo.wave = 4
        assert lfo.wave == "sine"


class TestOperator:
    """Tests for Operator class."""

    def test_operator_defaults(self):
        """Test operator default values."""
        op = Operator()
        assert op.output_level == 99
        assert op.frequency_coarse == 1
        assert op.frequency_mode == 0

    def test_operator_curves_by_name(self):
        """Test setting operator curves by name."""
        op = Operator()
        op.left_curve = "exp+"
        op.right_curve = "log"
        assert op.left_curve == "exp+"
        assert op.right_curve == "log"

    def test_operator_curves_by_index(self):
        """Test setting operator curves by index."""
        op = Operator()
        op.left_curve = 0
        op.right_curve = 3
        assert op.left_curve == "lin"
        assert op.right_curve == "log"

    def test_frequency_ratio(self):
        """Test frequency ratio calculation."""
        op = Operator()
        op.frequency_coarse = 2
        op.frequency_fine = 0
        op.frequency_mode = 0  # Ratio mode
        assert op.frequency_ratio == 2.0

    @pytest.mark.parametrize(
        "coarse,fine,expected_hz",
        [(0, 0, 1.0), (1, 0, 10.0), (2, 0, 100.0), (3, 0, 1000.0),
         (4, 0, 1.0), (5, 0, 10.0), (2, 30, 199.526231), (1, 50, 31.622777)],
    )
    def test_fixed_frequency_matches_engine_formula(self, coarse, fine, expected_hz):
        """Fixed mode is a decade per coarse unit, a centidecade per fine unit.

        The engine computes 10 ** ((coarse & 3) + fine / 100); only the low two
        bits of coarse are used, so coarse 4 wraps back to 1 Hz.
        """
        op = Operator()
        op.frequency_mode = 1
        op.frequency_coarse = coarse
        op.frequency_fine = fine
        assert op.frequency_ratio == pytest.approx(expected_hz)

    @pytest.mark.parametrize("coarse,fine", [(2, 0), (3, 0), (2, 30)])
    def test_fixed_frequency_matches_rendered_audio(self, coarse, fine):
        """The reported frequency must be the one the synth actually renders."""
        sample_rate = 44100.0
        patch = Patch()
        patch.algorithm = 31  # all six operators are carriers
        for i in range(6):
            patch.op[i].output_level = 99 if i == 0 else 0
            patch.op[i].envelope.rates = [99, 99, 99, 99]
            patch.op[i].envelope.levels = [99, 99, 99, 0]
        patch.op[0].frequency_mode = 1
        patch.op[0].frequency_coarse = coarse
        patch.op[0].frequency_fine = fine

        synth = DexedSynth(sample_rate=sample_rate)
        synth.load_patch(patch)
        audio = synth.render(
            midi_note=60, velocity=99, note_duration=0.5, render_duration=0.5
        )

        window = audio[2000:2000 + 16384]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        peak_hz = np.fft.rfftfreq(len(window), 1.0 / sample_rate)[np.argmax(spectrum)]

        bin_width = sample_rate / len(window)
        assert peak_hz == pytest.approx(patch.op[0].frequency_ratio, abs=2 * bin_width)


class TestBankIO:
    """Tests for reading and writing 32-voice bank files."""

    def test_save_to_bank_is_callable_on_the_class(self, tmp_path):
        """The documented call is Patch.save_to_bank(filename, patches).

        It used to be an instance method, so the documented form bound self to
        the filename and raised AttributeError without writing anything.
        """
        patches = [Patch(name=f"VOICE{i:02d}") for i in range(32)]
        path = tmp_path / "bank.syx"
        Patch.save_to_bank(str(path), patches)
        assert path.exists()

    def test_bank_round_trip_preserves_every_voice(self, tmp_path):
        patches = []
        for i in range(32):
            patch = Patch(name=f"VOICE{i:02d}")
            patch.algorithm = i
            patch.feedback = i % 8
            patches.append(patch)

        path = tmp_path / "bank.syx"
        Patch.save_to_bank(str(path), patches)
        loaded = Patch.load_bank(str(path))

        assert len(loaded) == 32
        for original, restored in zip(patches, loaded):
            assert restored.name.strip() == original.name.strip()
            assert restored.algorithm == original.algorithm
            assert restored.feedback == original.feedback

    def test_wrong_patch_count_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="exactly 32"):
            Patch.save_to_bank(str(tmp_path / "bank.syx"), [Patch()])

    def test_saved_bank_is_a_real_bulk_dump(self, tmp_path):
        """A .syx file has to be sendable to a DX7, not a bare payload."""
        path = tmp_path / "bank.syx"
        Patch.save_to_bank(str(path), [Patch() for _ in range(32)])

        data = path.read_bytes()
        assert len(data) == 4104
        assert data[:6] == bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])
        assert data[4103] == 0xF7
        assert data[4102] == (-sum(data[6:4102])) & 0x7F

    def test_raw_4096_byte_payload_still_loads(self, tmp_path):
        path = tmp_path / "bank.syx"
        Patch.save_to_bank(str(path), [Patch(name=f"V{i:02d}") for i in range(32)])
        raw = tmp_path / "raw.bin"
        raw.write_bytes(path.read_bytes()[6:4102])
        assert Patch.load_bank(str(raw))[3].name.strip() == "V03"

    def test_bank_preceded_by_another_sysex_message_is_found(self, tmp_path):
        """A bank used to be sliced blindly at [6:4102] and decoded as garbage."""
        path = tmp_path / "bank.syx"
        Patch.save_to_bank(str(path), [Patch(name=f"V{i:02d}") for i in range(32)])
        prefixed = tmp_path / "prefixed.syx"
        inquiry = bytes([0xF0, 0x7E, 0x00, 0x06, 0x02, 0xF7])
        prefixed.write_bytes(inquiry + path.read_bytes())

        assert Patch.load_bank(str(prefixed))[3].name.strip() == "V03"

    def test_bad_checksum_warns_but_still_loads(self, tmp_path):
        path = tmp_path / "bank.syx"
        Patch.save_to_bank(str(path), [Patch(name="VOICE") for _ in range(32)])
        data = bytearray(path.read_bytes())
        data[4102] ^= 0x7F
        path.write_bytes(bytes(data))

        with pytest.warns(UserWarning, match="checksum mismatch"):
            patches = Patch.load_bank(str(path))
        assert patches[0].name.strip() == "VOICE"

    def test_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "empty.syx"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            Patch.load_bank(str(path))

    def test_unrelated_sysex_is_rejected(self, tmp_path):
        path = tmp_path / "junk.syx"
        path.write_bytes(bytes([0xF0]) + bytes(5000))
        with pytest.raises(ValueError, match="No 32-voice DX7 bulk dump"):
            Patch.load_bank(str(path))


class TestSysexConversion:
    """Tests for sysex format conversion."""

    @pytest.mark.parametrize(
        "field,value,index,expected",
        [("algorithm", 32, 134, 31), ("algorithm", -1, 134, 0),
         ("feedback", 8, 135, 7), ("pitch_mod_sensitivity", 9, 143, 7),
         ("transpose", 60, 144, 48)],
    )
    def test_out_of_range_globals_saturate(self, field, value, index, expected):
        """An out-of-range value must saturate, not wrap around to zero.

        These fields used to be written with a bit mask, so algorithm 32 came
        out as algorithm 0 and maximum feedback came out as none at all.
        """
        patch = Patch()
        setattr(patch, field, value)
        assert patch.to_sysex()[index] == expected

    @pytest.mark.parametrize(
        "field,value,offset,expected",
        [("rate_scaling", 8, 13, 7), ("amp_mod_sensitivity", 4, 14, 3),
         ("velocity_sensitivity", 8, 15, 7), ("frequency_coarse", 32, 18, 31),
         ("detune", 15, 20, 14)],
    )
    def test_out_of_range_operator_fields_saturate(self, field, value, offset, expected):
        patch = Patch()
        setattr(patch.op[0], field, value)
        # op 0 is DX7 OP1, which sysex stores last
        assert patch.to_sysex()[5 * 21 + offset] == expected

    def test_out_of_range_algorithm_reaches_the_synth_as_31(self):
        patch = Patch()
        patch.algorithm = 32
        synth = DexedSynth()
        synth.load_patch(patch)
        assert synth.algorithm == 31

    def test_to_sysex_length(self):
        """Test that to_sysex returns 156 bytes."""
        patch = Patch()
        sysex = patch.to_sysex()
        assert len(sysex) == 156
        assert isinstance(sysex, bytes)

    def test_from_sysex_roundtrip(self):
        """Test sysex round-trip conversion."""
        patch = Patch(name="TestPatch")
        patch.algorithm = 15
        patch.feedback = 5
        patch.op[0].output_level = 80
        patch.op[2].frequency_coarse = 4

        sysex = patch.to_sysex()
        restored = Patch.from_sysex(sysex)

        assert restored.name.strip() == "TestPatch"
        assert restored.algorithm == 15
        assert restored.feedback == 5
        assert restored.op[0].output_level == 80
        assert restored.op[2].frequency_coarse == 4

    def test_to_packed_length(self):
        """Test that to_packed returns 128 bytes."""
        patch = Patch()
        packed = patch.to_packed()
        assert len(packed) == 128
        assert isinstance(packed, bytes)

    def test_from_packed_roundtrip(self):
        """Test packed format round-trip conversion."""
        patch = Patch(name="PackedTest")
        patch.algorithm = 7
        patch.op[1].output_level = 70

        packed = patch.to_packed()
        restored = Patch.from_packed(packed)

        assert restored.name.strip() == "PackedTest"
        assert restored.algorithm == 7
        assert restored.op[1].output_level == 70


class TestArrayConversion:
    """Tests for array format conversion."""

    def test_to_raw_shape(self):
        """Test that to_raw returns correct shape."""
        patch = Patch()
        raw = patch.to_raw()
        assert raw.shape == (155,)
        assert raw.dtype == np.uint8

    def test_from_raw_roundtrip(self):
        """Test raw format round-trip conversion."""
        patch = Patch()
        patch.algorithm = 19
        patch.op[3].output_level = 60

        raw = patch.to_raw()
        restored = Patch.from_raw(raw)

        assert restored.algorithm == 19
        assert restored.op[3].output_level == 60


class TestSynthWithPatch:
    """Tests for DexedSynth with Patch objects."""

    def test_load_patch(self):
        """Test loading a Patch into the synth."""
        synth = DexedSynth()
        patch = Patch()
        patch.algorithm = 15
        patch.op[0].output_level = 99

        synth.load_patch(patch)
        assert synth.algorithm == 15

    def test_render_with_patch(self):
        """Test rendering audio with a Patch."""
        synth = DexedSynth(sample_rate=44100)
        patch = Patch()
        patch.algorithm = 0
        patch.op[0].output_level = 99
        patch.op[0].envelope.rates = [99, 99, 99, 99]
        patch.op[0].envelope.levels = [99, 99, 99, 0]

        synth.load_patch(patch)
        audio = synth.render(midi_note=60, velocity=100,
                            note_duration=0.1, render_duration=0.2)

        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(44100 * 0.2)

    def test_algorithms_access(self):
        """Test accessing algorithms by number."""
        for i in range(32):
            alg = algorithms[i]
            assert alg.number == i
            assert isinstance(alg.carriers, list)
            assert isinstance(alg.modulators, list)

    def test_algorithms_invalid(self):
        """Test that invalid algorithm numbers raise errors."""
        with pytest.raises(KeyError):
            _ = algorithms[-1]
        with pytest.raises(KeyError):
            _ = algorithms[32]

    def test_algorithm_carriers(self):
        """Test carrier lists for specific algorithms."""
        # Algorithm 0: carriers are ops 0 and 2
        assert 0 in algorithms[0].carriers
        assert 2 in algorithms[0].carriers

        # Algorithm 31: all operators are carriers
        assert algorithms[31].carriers == [0, 1, 2, 3, 4, 5]
        assert algorithms[31].modulators == []

    def test_mod_matrix_shape(self):
        """Test modulation matrix shape."""
        for i in range(32):
            matrix = algorithms[i].mod_matrix
            assert matrix.shape == (6, 6)

    def test_get_carriers_function(self):
        """Test get_carriers helper function."""
        carriers = get_carriers(15)
        assert isinstance(carriers, list)
        assert 0 in carriers  # Algorithm 15 has op0 as carrier

    def test_get_modulators_function(self):
        """Test get_modulators helper function."""
        modulators = get_modulators(31)
        assert modulators == []  # Algorithm 31 has no modulators

    def test_get_mod_matrix_function(self):
        """Test get_mod_matrix helper function."""
        matrix = get_mod_matrix(0)
        assert matrix.shape == (6, 6)
        # get_mod_matrix returns a copy
        matrix[0, 0] = 99
        assert algorithms[0].mod_matrix[0, 0] != 99


class TestNormalizeFeedback:
    """Tests for feedback normalization feature."""

    def _create_feedback_patch(self, algorithm: int) -> Patch:
        """Create a patch with maximum feedback for testing."""
        patch = Patch()
        patch.algorithm = algorithm
        patch.feedback = 7  # Maximum feedback
        patch.op[0].output_level = 99
        patch.op[0].envelope.rates = [99, 99, 99, 99]
        patch.op[0].envelope.levels = [99, 99, 99, 0]
        return patch

    def test_normalize_feedback_default(self):
        """Test that normalize_feedback defaults to False."""
        synth = DexedSynth()
        assert synth.normalize_feedback is False

    def test_normalize_feedback_setter(self):
        """Test setting normalize_feedback flag."""
        synth = DexedSynth()
        synth.normalize_feedback = True
        assert synth.normalize_feedback is True
        synth.normalize_feedback = False
        assert synth.normalize_feedback is False

    def test_algorithm_1_unchanged(self):
        """Test that algorithm 0 is unchanged by normalization flag."""
        patch = self._create_feedback_patch(algorithm=0)
        synth = DexedSynth()

        synth.normalize_feedback = False
        synth.load_patch(patch)
        audio_authentic = synth.render(midi_note=60, velocity=100,
                                       note_duration=0.1, render_duration=0.15)

        synth.normalize_feedback = True
        audio_normalized = synth.render(midi_note=60, velocity=100,
                                        note_duration=0.1, render_duration=0.15)

        # Algorithm 0 should be identical
        diff = np.abs(audio_normalized - audio_authentic).max()
        assert diff < 1e-6, f"Algorithm 0 should be unchanged, diff={diff}"

    def test_algorithm_4_different(self):
        """Test that algorithm 3 differs with normalization flag."""
        patch = self._create_feedback_patch(algorithm=3)
        synth = DexedSynth()

        synth.normalize_feedback = False
        synth.load_patch(patch)
        audio_authentic = synth.render(midi_note=60, velocity=100,
                                       note_duration=0.1, render_duration=0.15)

        synth.normalize_feedback = True
        audio_normalized = synth.render(midi_note=60, velocity=100,
                                        note_duration=0.1, render_duration=0.15)

        # Algorithm 3 should be different
        diff = np.abs(audio_normalized - audio_authentic).max()
        assert diff > 0.01, f"Algorithm 3 should differ with normalization, diff={diff}"

    def test_algorithm_6_different(self):
        """Test that algorithm 5 differs with normalization flag."""
        patch = self._create_feedback_patch(algorithm=5)
        synth = DexedSynth()

        synth.normalize_feedback = False
        synth.load_patch(patch)
        audio_authentic = synth.render(midi_note=60, velocity=100,
                                       note_duration=0.1, render_duration=0.15)

        synth.normalize_feedback = True
        audio_normalized = synth.render(midi_note=60, velocity=100,
                                        note_duration=0.1, render_duration=0.15)

        # Algorithm 5 should be different
        diff = np.abs(audio_normalized - audio_authentic).max()
        assert diff > 0.01, f"Algorithm 5 should differ with normalization, diff={diff}"

    def test_algorithm_32_different(self):
        """Test that algorithm 31 differs with normalization flag."""
        patch = self._create_feedback_patch(algorithm=31)
        synth = DexedSynth()

        synth.normalize_feedback = False
        synth.load_patch(patch)
        audio_authentic = synth.render(midi_note=60, velocity=100,
                                       note_duration=0.1, render_duration=0.15)

        synth.normalize_feedback = True
        audio_normalized = synth.render(midi_note=60, velocity=100,
                                        note_duration=0.1, render_duration=0.15)

        # Algorithm 31 should be different
        diff = np.abs(audio_normalized - audio_authentic).max()
        assert diff > 0.01, f"Algorithm 31 should differ with normalization, diff={diff}"

    def test_normalize_feedback_pickle(self):
        """Test that normalize_feedback is preserved through pickle."""
        import pickle

        synth = DexedSynth()
        synth.normalize_feedback = True

        # Pickle and restore
        pickled = pickle.dumps(synth)
        restored = pickle.loads(pickled)

        assert restored.normalize_feedback is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
