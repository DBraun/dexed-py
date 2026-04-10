#!/usr/bin/env python
"""Tests for the OperatorGraph class."""

import numpy as np
import pytest
from dexed import OperatorGraph, GraphOperator, GraphEnvelope, algorithms


class TestOperatorGraphCreation:
    """Tests for OperatorGraph creation."""

    def test_create_default_graph(self):
        """Test creating a graph with default 6 operators."""
        graph = OperatorGraph()
        assert graph.num_ops == 6
        assert graph.carriers == []
        assert graph.modulators == [0, 1, 2, 3, 4, 5]

    def test_create_custom_size(self):
        """Test creating a graph with custom number of operators."""
        graph = OperatorGraph(num_ops=7)
        assert graph.num_ops == 7

        graph = OperatorGraph(num_ops=12)
        assert graph.num_ops == 12

    def test_create_invalid_size(self):
        """Test that invalid sizes raise errors."""
        with pytest.raises(ValueError):
            OperatorGraph(num_ops=0)
        with pytest.raises(ValueError):
            OperatorGraph(num_ops=-1)

    def test_operator_access(self):
        """Test 0-indexed operator access."""
        graph = OperatorGraph(num_ops=7)

        # Valid indices
        for i in range(7):
            op = graph.op[i]
            assert isinstance(op, GraphOperator)

        # Invalid indices
        with pytest.raises(IndexError):
            _ = graph.op[-1]
        with pytest.raises(IndexError):
            _ = graph.op[7]


class TestOperatorGraphConnections:
    """Tests for operator connections."""

    def test_connect_operators(self):
        """Test connecting operators."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(1, 0)  # Op1 modulates Op0

        matrix = graph.mod_matrix
        assert matrix[0, 1] == 1.0  # target 0, source 1

    def test_connect_with_amount(self):
        """Test connecting with custom modulation amount."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(1, 0, amount=0.5)

        matrix = graph.mod_matrix
        assert matrix[0, 1] == 0.5

    def test_disconnect(self):
        """Test disconnecting operators."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(1, 0)
        assert graph.mod_matrix[0, 1] == 1.0

        graph.disconnect(1, 0)
        assert graph.mod_matrix[0, 1] == 0.0

    def test_invalid_connections(self):
        """Test that invalid connections raise errors."""
        graph = OperatorGraph(num_ops=4)

        with pytest.raises(ValueError):
            graph.connect(-1, 0)  # Invalid source
        with pytest.raises(ValueError):
            graph.connect(4, 0)  # Source out of range
        with pytest.raises(ValueError):
            graph.connect(0, -1)  # Invalid target


class TestCarriersAndFeedback:
    """Tests for carrier and feedback settings."""

    def test_set_carriers(self):
        """Test setting carriers."""
        graph = OperatorGraph(num_ops=6)
        graph.set_carriers([0, 2, 4])

        assert graph.carriers == [0, 2, 4]
        assert set(graph.modulators) == {1, 3, 5}

    def test_invalid_carriers(self):
        """Test that invalid carriers raise errors."""
        graph = OperatorGraph(num_ops=4)

        with pytest.raises(ValueError):
            graph.set_carriers([-1])
        with pytest.raises(ValueError):
            graph.set_carriers([4])

    def test_set_feedback(self):
        """Test setting self-feedback."""
        graph = OperatorGraph(num_ops=4)
        graph.set_feedback(3, 3, level=7)

        assert graph._feedback.get((3, 3)) == 7

    def test_remove_feedback(self):
        """Test removing feedback by setting level to 0."""
        graph = OperatorGraph(num_ops=4)
        graph.set_feedback(3, 3, level=7)
        graph.set_feedback(3, 3, level=0)

        assert (3, 3) not in graph._feedback

    def test_set_cross_operator_feedback(self):
        """Test feedback from one operator to another."""
        graph = OperatorGraph(num_ops=6)
        graph.set_feedback(3, 5, level=5)

        assert graph._feedback.get((3, 5)) == 5
        assert graph.get_feedback(3, 5) == 5
        assert graph.get_feedback(3, 3) == 0


class TestGraphFromAlgorithm:
    """Tests for creating graphs from DX7 algorithms."""

    def test_from_algorithm_1(self):
        """Test creating graph from algorithm 0."""
        graph = OperatorGraph.from_algorithm(0)

        assert graph.num_ops == 6
        assert set(graph.carriers) == {0, 2}
        assert set(graph.modulators) == {1, 3, 4, 5}

    def test_from_algorithm_32(self):
        """Test creating graph from algorithm 31 (all carriers)."""
        graph = OperatorGraph.from_algorithm(31)

        assert graph.num_ops == 6
        assert set(graph.carriers) == {0, 1, 2, 3, 4, 5}
        assert graph.modulators == []

    def test_from_all_algorithms(self):
        """Test creating graphs from all 32 algorithms."""
        for alg_num in range(32):
            graph = OperatorGraph.from_algorithm(alg_num)
            alg = algorithms[alg_num]

            assert graph.num_ops == 6
            assert set(graph.carriers) == set(alg.carriers)


class TestGraphFromMatrix:
    """Tests for creating graphs from modulation matrices."""

    def test_from_matrix_basic(self):
        """Test creating graph from a modulation matrix."""
        mod_matrix = np.array([
            [0, 1, 0],  # Op1 modulates Op0
            [0, 0, 1],  # Op2 modulates Op1
            [0, 0, 0],
        ], dtype=np.float32)

        graph = OperatorGraph.from_matrix(mod_matrix, carriers=[0])

        assert graph.num_ops == 3
        assert graph.carriers == [0]
        np.testing.assert_array_equal(graph.mod_matrix, mod_matrix)

    def test_from_matrix_with_feedback(self):
        """Test creating graph from matrix with feedback."""
        mod_matrix = np.zeros((4, 4), dtype=np.float32)
        graph = OperatorGraph.from_matrix(
            mod_matrix, carriers=[0, 1], feedback={(3, 3): 6}
        )

        assert graph._feedback.get((3, 3)) == 6

    def test_from_matrix_invalid(self):
        """Test that invalid matrices raise errors."""
        # Non-square matrix
        with pytest.raises(ValueError):
            OperatorGraph.from_matrix(np.zeros((3, 4)), carriers=[0])


class TestRendering:
    """Tests for audio rendering."""

    def test_render_basic(self):
        """Test basic rendering."""
        graph = OperatorGraph(num_ops=2)
        graph.connect(1, 0)
        graph.set_carriers([0])

        audio = graph.render(
            sample_rate=44100,
            midi_note=60,
            velocity=100,
            note_duration=0.1,
            render_duration=0.15
        )

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == int(44100 * 0.15)

    def test_render_no_carriers_raises(self):
        """Test that rendering without carriers raises error."""
        graph = OperatorGraph(num_ops=2)

        with pytest.raises(ValueError, match="No carriers set"):
            graph.render()

    def test_render_all_ops_shape(self):
        """Test render_all_ops output shape."""
        graph = OperatorGraph(num_ops=7)
        for i in range(6):
            graph.connect(i + 1, i)
        graph.set_carriers([0])

        audio = graph.render_all_ops(
            sample_rate=44100,
            midi_note=60,
            note_duration=0.1,
            render_duration=0.15
        )

        # 7 operators + 1 final channel
        assert audio.shape == (8, int(44100 * 0.15))

    def test_render_different_notes(self):
        """Test rendering different MIDI notes produces different audio."""
        graph = OperatorGraph.from_algorithm(0)

        audio_60 = graph.render(midi_note=60, note_duration=0.1, render_duration=0.15)
        audio_72 = graph.render(midi_note=72, note_duration=0.1, render_duration=0.15)

        # Different notes should produce different audio
        assert not np.allclose(audio_60, audio_72)


class TestSevenOperators:
    """Tests specifically for 7+ operator graphs."""

    def test_seven_operator_cascade(self):
        """Test a 7-operator cascade: 6->5->4->3->2->1->0."""
        graph = OperatorGraph(num_ops=7)

        # Configure all operators
        for i in range(7):
            graph.op[i].output_level = 99
            graph.op[i].frequency_coarse = i + 1

        # Create cascade
        for i in range(5, -1, -1):
            graph.connect(i + 1, i)

        graph.set_carriers([0])
        graph.set_feedback(6, 6, level=5)

        audio = graph.render(
            sample_rate=44100,
            midi_note=60,
            velocity=100,
            note_duration=0.5,
            render_duration=0.6
        )

        assert len(audio) == int(44100 * 0.6)
        assert audio.max() > 0.1  # Should have some output

    def test_twelve_operator_graph(self):
        """Test a 12-operator graph (two 6-op stacks in parallel)."""
        graph = OperatorGraph(num_ops=12)

        # Stack 1: 5->4->3->2->1->0
        for i in range(4, -1, -1):
            graph.connect(i + 1, i)

        # Stack 2: 11->10->9->8->7->6
        for i in range(10, 5, -1):
            graph.connect(i + 1, i)

        # Both stack outputs are carriers
        graph.set_carriers([0, 6])

        # Configure operators
        for i in range(12):
            graph.op[i].output_level = 99

        audio = graph.render(
            sample_rate=44100,
            midi_note=60,
            velocity=100,
            note_duration=0.3,
            render_duration=0.4
        )

        assert len(audio) == int(44100 * 0.4)


class TestMethodChaining:
    """Tests for fluent API method chaining."""

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        graph = (OperatorGraph(num_ops=4)
                 .connect(3, 2)
                 .connect(2, 1)
                 .connect(1, 0)
                 .set_carriers([0])
                 .set_feedback(3, 3, level=5))

        assert isinstance(graph, OperatorGraph)
        assert graph.carriers == [0]


class TestGraphOperator:
    """Tests for GraphOperator class."""

    def test_operator_defaults(self):
        """Test GraphOperator default values."""
        op = GraphOperator()
        assert op.output_level == 99
        assert op.frequency_coarse == 1
        assert op.frequency_fine == 0
        assert op.frequency_mode == 0
        assert op.detune == 7

    def test_frequency_ratio(self):
        """Test frequency ratio calculation."""
        op = GraphOperator()
        op.frequency_coarse = 2
        op.frequency_fine = 0
        assert op.frequency_ratio == 2.0

        op.frequency_fine = 50
        assert op.frequency_ratio == 3.0  # 2 * 1.5


class TestGraphEnvelope:
    """Tests for GraphEnvelope class."""

    def test_envelope_defaults(self):
        """Test GraphEnvelope default values."""
        env = GraphEnvelope()
        assert env.rates == [99, 99, 99, 99]
        assert env.levels == [99, 99, 99, 0]

    def test_envelope_custom(self):
        """Test GraphEnvelope with custom values."""
        env = GraphEnvelope(rates=[80, 60, 40, 20], levels=[99, 80, 60, 0])
        assert env.rates == [80, 60, 40, 20]
        assert env.levels == [99, 80, 60, 0]

    def test_envelope_invalid_length(self):
        """Test that invalid envelope lengths raise errors."""
        with pytest.raises(ValueError):
            GraphEnvelope(rates=[99, 99, 99])
        with pytest.raises(ValueError):
            GraphEnvelope(levels=[99, 99])


class TestQueryMethods:
    """Tests for query/introspection API."""

    def test_get_connections_empty(self):
        """Test get_connections on empty graph."""
        graph = OperatorGraph(num_ops=4)
        assert graph.get_connections() == []

    def test_get_connections(self):
        """Test get_connections returns all connections."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 2)
        graph.connect(2, 1)
        graph.connect(1, 0, amount=0.5)

        connections = graph.get_connections()
        assert len(connections) == 3
        assert (3, 2, 1.0) in connections
        assert (2, 1, 1.0) in connections
        assert (1, 0, 0.5) in connections

    def test_get_sources(self):
        """Test get_sources for a target operator."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(2, 0)
        graph.connect(3, 0, amount=0.7)

        sources = graph.get_sources(0)
        assert len(sources) == 2
        assert (2, 1.0) in sources
        # Check for (3, ~0.7) with float tolerance
        src3 = [amt for op, amt in sources if op == 3]
        assert len(src3) == 1
        assert src3[0] == pytest.approx(0.7)

    def test_get_sources_none(self):
        """Test get_sources when no sources."""
        graph = OperatorGraph(num_ops=4)
        assert graph.get_sources(0) == []

    def test_get_targets(self):
        """Test get_targets for a source operator."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 0)
        graph.connect(3, 1, amount=0.5)

        targets = graph.get_targets(3)
        assert len(targets) == 2
        assert (0, 1.0) in targets
        assert (1, 0.5) in targets

    def test_get_targets_none(self):
        """Test get_targets when no targets."""
        graph = OperatorGraph(num_ops=4)
        assert graph.get_targets(3) == []

    def test_is_connected(self):
        """Test is_connected check."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 2)

        assert graph.is_connected(3, 2) is True
        assert graph.is_connected(2, 3) is False
        assert graph.is_connected(0, 1) is False

    def test_connection_amount(self):
        """Test connection_amount retrieval."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 2, amount=0.75)

        assert graph.connection_amount(3, 2) == 0.75
        assert graph.connection_amount(2, 3) == 0.0
        assert graph.connection_amount(0, 1) == 0.0

    def test_get_feedback(self):
        """Test get_feedback retrieval."""
        graph = OperatorGraph(num_ops=4)
        graph.set_feedback(3, 3, level=5)

        assert graph.get_feedback(3, 3) == 5
        assert graph.get_feedback(2, 2) == 0
        assert graph.get_feedback(0, 0) == 0

    def test_disconnect_all(self):
        """Test disconnect_all removes all connections."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 2)
        graph.connect(2, 1)
        graph.connect(1, 0)

        assert len(graph.get_connections()) == 3
        graph.disconnect_all()
        assert len(graph.get_connections()) == 0

    def test_disconnect_all_chaining(self):
        """Test disconnect_all returns self for chaining."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 2)
        result = graph.disconnect_all()
        assert result is graph


class TestVisualization:
    """Tests for visualization/debug API."""

    def test_summary_basic(self):
        """Test summary output format."""
        graph = OperatorGraph(num_ops=4)
        graph.connect(3, 2)
        graph.connect(2, 1)
        graph.connect(1, 0)
        graph.set_carriers([0])
        graph.set_feedback(3, 3, level=5)

        summary = graph.summary()
        assert "OperatorGraph (4 operators)" in summary
        assert "Carriers: [0]" in summary
        assert "Modulators: [1, 2, 3]" in summary
        assert "Connections:" in summary
        assert "3 -> 2" in summary
        assert "Feedback:" in summary
        assert "Op 3: level 5" in summary

    def test_summary_no_connections(self):
        """Test summary with no connections."""
        graph = OperatorGraph(num_ops=2)
        graph.set_carriers([0])
        summary = graph.summary()
        assert "Connections: none" in summary

    def test_to_mermaid_basic(self):
        """Test Mermaid diagram generation."""
        graph = OperatorGraph(num_ops=3)
        graph.connect(2, 1)
        graph.connect(1, 0)
        graph.set_carriers([0])
        graph.set_feedback(2, 2, level=7)

        mermaid = graph.to_mermaid()
        assert "flowchart TB" in mermaid
        assert "op0[Op 0]:::carrier" in mermaid
        assert "op2 --> op1" in mermaid
        assert "op1 --> op0" in mermaid
        assert "op2 -.->|fb:7| op2" in mermaid
        assert "classDef carrier fill:#90EE90" in mermaid

    def test_to_mermaid_with_amounts(self):
        """Test Mermaid diagram shows non-1.0 amounts."""
        graph = OperatorGraph(num_ops=2)
        graph.connect(1, 0, amount=0.5)
        graph.set_carriers([0])

        mermaid = graph.to_mermaid()
        assert "op1 -->|0.5| op0" in mermaid

    def test_to_ascii_basic(self):
        """Test ASCII art generation."""
        graph = OperatorGraph(num_ops=3)
        graph.connect(2, 1)
        graph.connect(1, 0)
        graph.set_carriers([0])
        graph.set_feedback(2, 2, level=5)

        ascii_art = graph.to_ascii()
        assert "OperatorGraph (3 operators)" in ascii_art
        assert "[2]" in ascii_art
        assert "[1]" in ascii_art
        assert "[0]" in ascii_art
        assert "(C)" in ascii_art  # Carrier marker
        assert "Carriers: 0" in ascii_art
        assert "Feedback:" in ascii_art

    def test_to_ascii_no_connections(self):
        """Test ASCII art with isolated operators."""
        graph = OperatorGraph(num_ops=2)
        graph.set_carriers([0, 1])

        ascii_art = graph.to_ascii()
        assert "[0]" in ascii_art
        assert "[1]" in ascii_art
        assert "Carriers: 0, 1" in ascii_art


class TestSpectralComparison:
    """Tests comparing OperatorGraph spectral output with DexedSynth."""

    def _compute_spectrum(self, audio, sample_rate=44100):
        """Compute magnitude spectrum of audio."""
        # Apply window and compute FFT on full audio
        window = np.hanning(len(audio))
        spectrum = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

        return freqs, spectrum

    def _spectral_correlation(self, spec1, spec2):
        """Compute correlation between two spectra."""
        # Normalize spectra
        spec1 = spec1 / (np.max(spec1) + 1e-10)
        spec2 = spec2 / (np.max(spec2) + 1e-10)

        # Correlation
        s1 = spec1 - spec1.mean()
        s2 = spec2 - spec2.mean()
        denom = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
        if denom < 1e-10:
            return 0.0
        return np.sum(s1 * s2) / denom

    def _find_peaks(self, freqs, spectrum, threshold=0.1):
        """Find frequency peaks above threshold."""
        max_val = np.max(spectrum)
        peak_mask = spectrum > (threshold * max_val)
        peak_freqs = freqs[peak_mask]
        return peak_freqs

    def test_no_feedback_spectral_match(self):
        """Test that spectra match well without feedback."""
        from dexed import DexedSynth, Patch

        # Setup identical parameters
        patch = Patch()
        patch.algorithm = 0
        patch.feedback = 0
        for i in range(6):
            patch.op[i].output_level = 99
            patch.op[i].frequency_coarse = 1
            patch.op[i].frequency_fine = 0
            patch.op[i].detune = 7
            patch.op[i].envelope.rates = [99, 99, 99, 99]
            patch.op[i].envelope.levels = [99, 99, 99, 0]

        synth = DexedSynth(sample_rate=44100)
        synth.load_patch(patch)
        cpp_audio = synth.render(midi_note=60, velocity=100,
                                 note_duration=0.5, render_duration=0.5)

        graph = OperatorGraph.from_algorithm(0)
        for i in range(6):
            graph.op[i].output_level = 99
            graph.op[i].frequency_coarse = 1
            graph.op[i].frequency_fine = 0
            graph.op[i].detune = 7
            graph.op[i].envelope.rates = [99, 99, 99, 99]
            graph.op[i].envelope.levels = [99, 99, 99, 0]

        py_audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                                note_duration=0.5, render_duration=0.5)

        # Compare spectra
        _, cpp_spec = self._compute_spectrum(cpp_audio)
        _, py_spec = self._compute_spectrum(py_audio)

        corr = self._spectral_correlation(cpp_spec, py_spec)
        assert corr > 0.95, f"Spectral correlation {corr:.4f} should be > 0.95 without feedback"

    def test_feedback_spectral_similarity(self):
        """Test that spectra have similar characteristics with feedback."""
        from dexed import DexedSynth, Patch

        # Setup with feedback
        patch = Patch()
        patch.algorithm = 31  # All carriers
        patch.feedback = 7  # Max feedback
        for i in range(6):
            patch.op[i].output_level = 0
            patch.op[i].frequency_coarse = 1
            patch.op[i].frequency_fine = 0
            patch.op[i].detune = 7
            patch.op[i].envelope.rates = [99, 99, 99, 99]
            patch.op[i].envelope.levels = [99, 99, 99, 0]
        patch.op[5].output_level = 99  # Only feedback operator active

        synth = DexedSynth(sample_rate=44100)
        synth.normalize_feedback = True  # Use same feedback formula
        synth.load_patch(patch)
        cpp_audio = synth.render(midi_note=60, velocity=100,
                                 note_duration=0.5, render_duration=0.5)

        graph = OperatorGraph.from_algorithm(31)
        for i in range(6):
            graph.op[i].output_level = 0
            graph.op[i].frequency_coarse = 1
            graph.op[i].frequency_fine = 0
            graph.op[i].detune = 7
            graph.op[i].envelope.rates = [99, 99, 99, 99]
            graph.op[i].envelope.levels = [99, 99, 99, 0]
        graph.op[5].output_level = 99
        graph.set_feedback(5, 5, level=7)

        py_audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                                note_duration=0.5, render_duration=0.5)

        # Compute spectra
        freqs, cpp_spec = self._compute_spectrum(cpp_audio)
        _, py_spec = self._compute_spectrum(py_audio)

        # With high feedback, both should have energy spread across frequencies
        # Check that high-frequency energy (>1kHz) is a significant portion of total
        freq_1k_idx = np.searchsorted(freqs, 1000)
        cpp_hf_ratio = np.sum(cpp_spec[freq_1k_idx:]**2) / np.sum(cpp_spec**2)
        py_hf_ratio = np.sum(py_spec[freq_1k_idx:]**2) / np.sum(py_spec**2)

        # Both should have at least 1% of energy above 1kHz (feedback creates harmonics)
        assert cpp_hf_ratio > 0.01, f"C++ high-freq ratio {cpp_hf_ratio:.4f} should be > 0.01"
        assert py_hf_ratio > 0.01, f"Python high-freq ratio {py_hf_ratio:.4f} should be > 0.01"

        # RMS should be in similar range (both producing sound)
        cpp_rms = np.sqrt(np.mean(cpp_audio[2000:20000]**2))
        py_rms = np.sqrt(np.mean(py_audio[2000:20000]**2))

        rms_ratio = min(cpp_rms, py_rms) / max(cpp_rms, py_rms)
        assert rms_ratio > 0.8, f"RMS ratio {rms_ratio:.4f} should be > 0.8"

    def test_feedback_produces_harmonics(self):
        """Test that feedback creates rich harmonic content in both implementations."""
        from dexed import DexedSynth, Patch

        # Without feedback - should be pure sine
        patch_no_fb = Patch()
        patch_no_fb.algorithm = 31
        patch_no_fb.feedback = 0
        for i in range(6):
            patch_no_fb.op[i].output_level = 0
            patch_no_fb.op[i].frequency_coarse = 1
            patch_no_fb.op[i].envelope.rates = [99, 99, 99, 99]
            patch_no_fb.op[i].envelope.levels = [99, 99, 99, 0]
        patch_no_fb.op[5].output_level = 99

        graph_no_fb = OperatorGraph.from_algorithm(31)
        for i in range(6):
            graph_no_fb.op[i].output_level = 0
            graph_no_fb.op[i].frequency_coarse = 1
            graph_no_fb.op[i].envelope.rates = [99, 99, 99, 99]
            graph_no_fb.op[i].envelope.levels = [99, 99, 99, 0]
        graph_no_fb.op[5].output_level = 99

        py_no_fb = graph_no_fb.render(sample_rate=44100, midi_note=60, velocity=100,
                                       note_duration=0.5, render_duration=0.5)

        # With feedback - should have harmonics
        graph_fb = OperatorGraph.from_algorithm(31)
        for i in range(6):
            graph_fb.op[i].output_level = 0
            graph_fb.op[i].frequency_coarse = 1
            graph_fb.op[i].envelope.rates = [99, 99, 99, 99]
            graph_fb.op[i].envelope.levels = [99, 99, 99, 0]
        graph_fb.op[5].output_level = 99
        graph_fb.set_feedback(5, 5, level=7)

        py_fb = graph_fb.render(sample_rate=44100, midi_note=60, velocity=100,
                                 note_duration=0.5, render_duration=0.5)

        # Compare spectral complexity
        freqs, spec_no_fb = self._compute_spectrum(py_no_fb)
        _, spec_fb = self._compute_spectrum(py_fb)

        # Count significant frequency bins
        threshold = 0.1
        bins_no_fb = np.sum(spec_no_fb > threshold * np.max(spec_no_fb))
        bins_fb = np.sum(spec_fb > threshold * np.max(spec_fb))

        # Feedback should create more spectral content
        assert bins_fb > bins_no_fb * 2, \
            f"Feedback should create more harmonics: {bins_fb} vs {bins_no_fb}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
