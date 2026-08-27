#!/usr/bin/env python
"""Tests that _ALGORITHM_DATA agrees with the engine it describes.

The table in dexed/algorithms.py is a hand-maintained copy of the operator
routing table the C++ core actually renders. Nothing used to check the two
against each other, which is how algorithms 18 and 21 came to claim feedback on
operator 6 when the engine puts it on operator 3.

Two independent checks here. The first decodes FmCore::algorithms[32] out of the
C++ source and compares it field by field. The second renders audio and watches
where the feedback control goes inert; it builds its prediction from
dexed/algorithms.py alone, so a mistake in the decoder cannot excuse a mistake
in the table.
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


def test_cpp_source_is_present():
    """The comparison tests are worthless if the source silently goes missing.

    tests/ ships only in a checkout or an sdist, and MANIFEST.in puts src/*.cc
    in the sdist, so a missing fm_core.cc means a broken tree -- not a mode this
    suite is expected to run in. Failing here beats skipping 128 tests and
    reporting green.
    """
    assert FM_CORE.exists(), f"vendored engine source not found at {FM_CORE}"


@pytest.fixture(scope="module")
def cpp_table():
    return _decode_cpp_table()


@pytest.mark.parametrize("alg", range(32))
def test_carriers_match_cpp(cpp_table, alg):
    """Carriers agree with the operators the C++ table writes to the output."""
    expected = cpp_table[alg]["carriers"]
    assert sorted(c + 1 for c in algorithms[alg].carriers) == expected


@pytest.mark.parametrize("alg", range(32))
def test_mod_matrix_matches_cpp(cpp_table, alg):
    """Modulation edges agree with the buses the C++ table routes between."""
    matrix = algorithms[alg].mod_matrix
    edges = sorted(
        (j + 1, i + 1) for i in range(6) for j in range(6) if matrix[i][j]
    )
    assert edges == cpp_table[alg]["edges"]


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


def test_get_feedback_edge_is_exported_and_agrees_with_the_table():
    """The 0.2.1 changelog advertises this export; nothing used it."""
    import dexed

    assert "get_feedback_edge" in dexed.__all__
    for alg in range(32):
        assert dexed.get_feedback_edge(alg) == algorithms[alg].feedback_edge


@pytest.mark.parametrize(
    "alg, expected",
    [(0, (5, 5)), (17, (2, 2)), (3, (3, 5)), (5, (4, 5))],
)
def test_documented_feedback_edge_examples(alg, expected):
    """The exact values README.md and docs/quickstart.md print."""
    import dexed

    assert dexed.get_feedback_edge(alg) == expected


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


def _table_edges(alg):
    """Modulation edges of `alg` as (source, target), from the Python table."""
    matrix = algorithms[alg].mod_matrix
    return [(j, i) for i in range(6) for j in range(6) if matrix[i][j]]


def _reaches_output(alg, start, without):
    """Can `start`'s signal still reach a carrier with operator `without` muted?"""
    if start == without:
        return False
    carriers = set(algorithms[alg].carriers)
    edges = _table_edges(alg)
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        if node in carriers:
            return True
        for mod, car in edges:
            if mod == node and car != without and car not in seen:
                seen.add(car)
                stack.append(car)
    return False


def _collapsed_chain_interior(alg, src, tgt):
    """Operators inside a collapsed feedback chain, excluding both ends.

    EngineMkI fuses the multi-operator feedback loops of DX7 algorithms 4 and 6
    into a single sample loop (compute_fb3 / compute_fb2) and applies no level
    threshold to the operators inside it, unlike the normal render path. Muting
    one of those therefore does not stop feedback the way the routing graph says
    it should -- it leaves a residue around -23 dB -- so the reachability model
    cannot classify it either way and it is left out.

    Only algorithm 4 has an interior; algorithm 6's chain is two operators long.
    """
    if src == tgt:
        return set()
    edges = _table_edges(alg)
    stack = [(tgt, [tgt])]
    while stack:
        node, path = stack.pop()
        if node == src:
            return set(path[1:-1])
        for mod, car in edges:
            if mod == node and car not in path:
                stack.append((car, path + [car]))
    return set()


@pytest.mark.parametrize("alg", range(32))
def test_feedback_edge_matches_engine_behaviour(alg):
    """The engine's feedback control goes inert exactly where the table predicts.

    Muting one operator at a time, the feedback amount stops affecting the audio
    precisely when the mute silences the feedback source or cuts every path from
    the operator reading the feedback buffer to a carrier. The prediction comes
    from dexed/algorithms.py and the measurement from rendered audio; the C++
    source is not consulted, so this stands on its own if the decoder above is
    ever wrong.
    """
    src, tgt = algorithms[alg].feedback_edge

    deltas, levels = [], []
    for op in range(6):
        dry, wet = _render(alg, 0, op), _render(alg, 7, op)
        deltas.append(np.sqrt(np.mean(np.square(wet - dry))))
        levels.append(np.sqrt(np.mean(np.square(wet))))

    peak = max(deltas)
    assert peak > 0, "feedback had no audible effect on any operator"

    # A mute that silences the patch outright tells us nothing, and so does one
    # inside a collapsed feedback chain.
    collapsed = _collapsed_chain_interior(alg, src, tgt)
    assert not collapsed or alg == 3, (
        f"unexpected collapsed chain interior {sorted(collapsed)} for algorithm "
        f"{alg}; only DX7 algorithm 4 should have one"
    )
    classified = [
        op for op in range(6) if levels[op] > 1e-4 and op not in collapsed
    ]

    # Every point must land firmly on one side. A ratio in between means the
    # model and the engine disagree about something -- fail loudly rather than
    # bucketing it by whichever side of a single threshold it happens to fall.
    for op in classified:
        ratio = deltas[op] / peak
        assert ratio < 1e-3 or ratio > 0.5, (
            f"algorithm {alg}, muting operator {op}: feedback delta ratio "
            f"{ratio:.5f} is neither inert nor live"
        )

    inert = {op for op in classified if deltas[op] / peak < 1e-3}
    expected = {
        op for op in classified
        if op == src or not _reaches_output(alg, tgt, op)
    }
    assert inert == expected
