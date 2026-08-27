"""Concurrent use of a single synth."""

import threading

import numpy as np
import pytest

from dexed import DexedSynth, Patch


def _patch():
    patch = Patch()
    patch.algorithm = 7
    patch.feedback = 5
    return patch


def test_shared_synth_renders_from_many_threads():
    """A synth shared across threads must not corrupt its own state.

    render() swaps out the per-instance voice and LFO and then computes with
    the GIL released, so without serialization one thread frees the note
    another is still rendering through and the interpreter dies.
    """
    synth = DexedSynth(sample_rate=44100.0)
    synth.load_patch(_patch())

    expected = synth.render(
        midi_note=60, velocity=100, note_duration=0.05, render_duration=0.1
    )
    results = []
    errors = []

    def worker():
        try:
            for _ in range(25):
                results.append(
                    synth.render(
                        midi_note=60,
                        velocity=100,
                        note_duration=0.05,
                        render_duration=0.1,
                    )
                )
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 100
    for audio in results:
        assert np.array_equal(audio, expected)


def test_load_from_one_thread_while_another_renders():
    """Loading a patch mid-render must not be visible half-applied."""
    synth = DexedSynth(sample_rate=44100.0)
    synth.load_patch(_patch())

    other = Patch()
    other.algorithm = 31
    stop = threading.Event()
    errors = []

    def loader():
        try:
            while not stop.is_set():
                synth.load_patch(other)
                synth.load_patch(_patch())
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    thread = threading.Thread(target=loader)
    thread.start()
    try:
        for _ in range(25):
            audio = synth.render(
                midi_note=60, velocity=100, note_duration=0.05, render_duration=0.1
            )
            assert audio.shape == (4410,)
            assert np.all(np.isfinite(audio))
    finally:
        stop.set()
        thread.join()

    assert errors == []
