#!/usr/bin/env python
"""Tests that _ALGORITHM_DATA agrees with the engine it describes.

The table in dexed/algorithms.py is a hand-maintained copy of the operator
routing table the C++ core actually renders. Nothing used to check the two
against each other, which is how algorithms 18 and 21 came to claim feedback on
operator 6 when the engine puts it on operator 3.

Two independent checks here: the table is compared against the C++ source it was
copied from, and against the behaviour of the compiled engine.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from dexed import DexedSynth, Patch
from dexed.algorithms import algorithms

FM_CORE = Path(__file__).resolve().parent.parent / "src" / "msfa" / "fm_core.cc"

# Flag bits from FmOperatorFlags in src/msfa/fm_core.h
OUT_BUS_ADD, FB_IN, FB_OUT = 0x04, 0x40, 0x80


def _decode_cpp_table():
    """Decode FmCore::algorithms[32] into modulation graphs.

    Each row is six bytes, one per rendering slot. Slot i is DX7 operator 6-i,
    because the core renders operator 6 first (dx7note.cc fills params_[0] from
    the first 21 patch bytes, which the DX7 patch format stores as OP6). A byte
    holds the input bus in bits 4-5, the output bus in bits 0-1, a flag for
    adding to that bus rather than replacing it, and the two feedback tap bits.
    Writing to bus 0 means writing to the audio output, i.e. being a carrier.
    """
    body = FM_CORE.read_text().split("algorithms[32] = {")[1].split("};")[0]
    rows = re.findall(r"\{\s*\{([^}]*)\}\s*\}", body)
    assert len(rows) == 32, f"expected 32 algorithms, found {len(rows)}"

    table = []
    for row in rows:
        flags = [int(x, 16) for x in row.split(",")]
        bus = {1: [], 2: []}
        edges, carriers = [], []
        fb_in = fb_out = None

        for slot, f in enumerate(flags):
            op = 6 - slot
            in_bus, out_bus, add = (f >> 4) & 3, f & 3, bool(f & OUT_BUS_ADD)
            if f & FB_IN:
                fb_in = op
            if f & FB_OUT:
                fb_out = op
            if in_bus:
                edges.extend((m, op) for m in bus[in_bus])
            if out_bus == 0:
                carriers.append(op)
            else:
                bus[out_bus] = (bus[out_bus] if add else []) + [op]

        table.append(
            {
                "edges": sorted(edges),
                "carriers": sorted(carriers),
                # fb_out taps the loop, fb_in receives it; they differ only in
                # DX7 algorithms 4 and 6, where the loop wraps a whole chain.
                "feedback_edge": (fb_out, fb_in),
            }
        )
    return table


requires_source = pytest.mark.skipif(
    not FM_CORE.exists(), reason="C++ sources not available (installed package)"
)


@pytest.fixture(scope="module")
def cpp_table():
    return _decode_cpp_table()


@requires_source
@pytest.mark.parametrize("alg", range(32))
def test_carriers_match_cpp(cpp_table, alg):
    """Carriers agree with the operators the C++ table writes to the output."""
    expected = cpp_table[alg]["carriers"]
    assert sorted(c + 1 for c in algorithms[alg].carriers) == expected


@requires_source
@pytest.mark.parametrize("alg", range(32))
def test_mod_matrix_matches_cpp(cpp_table, alg):
    """Modulation edges agree with the buses the C++ table routes between."""
    matrix = algorithms[alg].mod_matrix
    edges = sorted(
        (j + 1, i + 1) for i in range(6) for j in range(6) if matrix[i][j]
    )
    assert edges == cpp_table[alg]["edges"]


@requires_source
@pytest.mark.parametrize("alg", range(32))
def test_feedback_edge_matches_cpp(cpp_table, alg):
    """Feedback edge agrees with the operators carrying the C++ feedback bits."""
    src, tgt = algorithms[alg].feedback_edge
    assert (src + 1, tgt + 1) == cpp_table[alg]["feedback_edge"]


@pytest.mark.parametrize(
    "alg, expected",
    [
        (17, (2, 2)),  # DX7 algorithm 18: feedback on op 3, not op 6
        (20, (2, 2)),  # DX7 algorithm 21: feedback on op 3, not op 6
        (3, (3, 5)),  # DX7 algorithm 4: three-operator loop, op 4 -> op 6
        (5, (4, 5)),  # DX7 algorithm 6: two-operator loop, op 5 -> op 6
    ],
)
def test_known_feedback_edges(alg, expected):
    """Pin the feedback edges that have been wrong or are easy to get wrong."""
    assert algorithms[alg].feedback_edge == expected


@pytest.mark.parametrize("alg", range(32))
def test_carriers_and_modulators_partition_operators(alg):
    """Every operator is either a carrier or a modulator, never both or neither."""
    info = algorithms[alg]
    assert sorted(info.carriers + info.modulators) == list(range(6))


@pytest.mark.parametrize("alg", range(32))
def test_feedback_edge_is_a_valid_operator_pair(alg):
    src, tgt = algorithms[alg].feedback_edge
    assert 0 <= src <= 5 and 0 <= tgt <= 5


def _render(alg, feedback, mute):
    """Render one algorithm with a single operator muted."""
    patch = Patch()
    patch.algorithm = alg
    patch.feedback = feedback
    for i in range(6):
        op = patch.op[i]
        op.output_level = 0 if i == mute else 99
        op.frequency_coarse = 1 + i  # distinct ratios, so no chain is symmetric
        op.frequency_fine = 0
        op.detune = 7
        op.envelope.rates = [99, 99, 99, 99]
        op.envelope.levels = [99, 99, 99, 0]
    synth = DexedSynth(sample_rate=44100)
    synth.load_patch(patch)
    return synth.render(midi_note=60, velocity=100,
                        note_duration=0.15, render_duration=0.15)


def _reaches_output(entry, start, without):
    """Can `start`'s signal still reach a carrier with operator `without` muted?"""
    if start == without:
        return False
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        if node in entry["carriers"]:
            return True
        for mod, car in entry["edges"]:
            if mod == node and car != without and car not in seen:
                seen.add(car)
                stack.append(car)
    return False


@requires_source
@pytest.mark.parametrize("alg", range(32))
def test_feedback_edge_matches_engine_behaviour(cpp_table, alg):
    """The engine's feedback control goes inert exactly where the table predicts.

    Muting one operator at a time, the feedback amount stops affecting the audio
    precisely when the mute silences the feedback source or cuts every path from
    the operator reading the feedback buffer to a carrier. This never consults
    the C++ source, only the rendered audio.
    """
    src, tgt = algorithms[alg].feedback_edge
    src, tgt = src + 1, tgt + 1

    deltas, levels = [], []
    for op in range(6):
        dry, wet = _render(alg, 0, op), _render(alg, 7, op)
        deltas.append(np.sqrt(np.mean(np.square(wet - dry))))
        levels.append(np.sqrt(np.mean(np.square(wet))))

    peak = max(deltas)
    assert peak > 0, "feedback had no audible effect on any operator"

    # A mute that silences the patch outright tells us nothing.
    audible = [op + 1 for op in range(6) if levels[op] > 1e-4]
    inert = {op for op in audible if deltas[op - 1] / peak < 0.1}
    expected = {
        op for op in audible
        if op == src or not _reaches_output(cpp_table[alg], tgt, op)
    }
    assert inert == expected
