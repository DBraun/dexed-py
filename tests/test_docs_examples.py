"""Executes the API forms the README and quickstart show.

Both documents drifted from the code during the 0.2.0 feedback rework -- the
`from_matrix(feedback={op: level})` examples kept the removed int-keyed dict and
raised TypeError when copied -- because nothing ran them.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from dexed import OperatorGraph, Patch

DOCS = [
    Path(__file__).resolve().parent.parent / "README.md",
    Path(__file__).resolve().parent.parent / "docs" / "quickstart.md",
]


def _snippets(pattern):
    """Every line across the docs matching `pattern`, with its source."""
    found = []
    for path in DOCS:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(pattern, line):
                found.append((path.name, number, line.strip()))
    return found


def test_documented_feedback_dicts_are_keyed_by_operator_pairs():
    """`feedback={op: level}` was removed in 0.2.0; it now raises TypeError."""
    lines = _snippets(r"feedback=\{")
    assert lines, "no from_matrix feedback examples found in the docs"
    for name, number, line in lines:
        assert re.search(r"feedback=\{\(\d+,\s*\d+\):", line), (
            f"{name}:{number} uses the removed int-keyed feedback dict: {line}"
        )


def test_documented_set_feedback_calls_pass_source_and_target():
    lines = _snippets(r"\.set_feedback\(")
    assert lines, "no set_feedback examples found in the docs"
    for name, number, line in lines:
        arguments = line.split(".set_feedback(", 1)[1].split(")")[0]
        assert arguments.count(",") >= 2, (
            f"{name}:{number} calls set_feedback with fewer than three "
            f"arguments: {line}"
        )


def test_from_matrix_feedback_example_runs():
    """The exact quickstart snippet."""
    mod_matrix = np.zeros((4, 4), dtype=np.float32)
    mod_matrix[0, 1] = 1.0
    mod_matrix[1, 2] = 1.0
    graph = OperatorGraph.from_matrix(
        mod_matrix, carriers=[0], feedback={(3, 3): 5}
    )
    assert graph.get_feedback(3, 3) == 5


def test_seven_operator_chain_example_runs():
    """The README's 7-operator chain, feedback included."""
    graph = OperatorGraph(num_ops=7)
    for i in range(7):
        graph.op[i].output_level = 99
    for i in range(6, 0, -1):
        graph.connect(i, i - 1)
    graph.set_carriers([0])
    graph.set_feedback(6, 6, level=7)

    audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                         note_duration=0.1, render_duration=0.15)
    assert audio.shape == (6615,)


def test_bank_save_and_load_example_runs(tmp_path):
    """`Patch.save_to_bank(filename, patches)`, exactly as documented."""
    patches = [Patch(name=f"VOICE{i:02d}") for i in range(32)]
    path = tmp_path / "my_bank.syx"
    Patch.save_to_bank(str(path), patches)
    assert Patch.load_bank(str(path))[0].name.strip() == "VOICE00"


def test_documented_bank_size_is_the_real_one():
    for name, number, line in _snippets(r"4096-byte `?\.syx"):
        pytest.fail(f"{name}:{number} calls a bank a 4096-byte .syx: {line}")


def test_version_metadata_agrees_everywhere():
    """CITATION.cff is the one version string not derived from version.py.

    It drifted the moment version.py was bumped, and nothing caught it: the
    "Cite this repository" button would have named the previous release.
    """
    import dexed

    root = Path(__file__).resolve().parent.parent
    citation = (root / "CITATION.cff").read_text()
    match = re.search(r'^version:\s*"([^"]+)"', citation, re.MULTILINE)
    assert match, "CITATION.cff has no version field"
    assert match.group(1) == dexed.__version__

    changelog = (root / "CHANGELOG.md").read_text()
    assert f"## [{dexed.__version__}]" in changelog, (
        f"CHANGELOG.md has no section for {dexed.__version__}"
    )


def test_license_metadata_agrees_everywhere():
    """pyproject.toml shipped Apache-2.0 for GPL-3.0 code from the first commit.

    The README sentence it came from scopes Apache 2.0 to src/msfa alone; the
    project itself, and both LICENSE files, are GPL. Nothing checked.
    """
    root = Path(__file__).resolve().parent.parent
    expected = "GPL-3.0-or-later"

    pyproject = (root / "pyproject.toml").read_text()
    match = re.search(r'^license\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject.toml has no license expression"
    assert match.group(1) == expected

    citation = (root / "CITATION.cff").read_text()
    match = re.search(r"^license:\s*(\S+)", citation, re.MULTILINE)
    assert match, "CITATION.cff has no license field"
    assert match.group(1) == expected

    # The bundled licence text is the GPL, not Apache.
    assert "GNU GENERAL PUBLIC LICENSE" in (root / "LICENSE").read_text()

    # PEP 639: a license expression and License:: classifiers are exclusive.
    assert "License :: OSI Approved" not in pyproject
