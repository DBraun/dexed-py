# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Sharing a `DexedSynth` between threads no longer crashes the interpreter.
  Loading and rendering are now serialized per instance.
- Creating a second `DexedSynth` at a different sample rate no longer retunes
  every synth already alive. The shared DX7 frequency, envelope and LFO tables
  are re-pointed at the rendering synth's own sample rate; rendering two
  different sample rates concurrently now raises instead of returning
  quietly detuned audio.
- `Operator.frequency_ratio` and `GraphOperator.frequency_ratio` now report the
  correct frequency in fixed-frequency mode. Both used a decade-times-multiplier
  formula that disagreed with the engine for every setting except
  `coarse=0, fine=0`, by as much as 250x. Rendering was always correct; only the
  reported value was wrong.
- `render_all_ops()` no longer returns silent channels for DX7 algorithms 4 and
  6 when feedback is non-zero. Those algorithms collapse their feedback chain
  into one loop, and the operators inside it were left at zero while the chain's
  output was attributed to the wrong channel, so `x[get_carriers(alg)].sum(0)`
  did not equal the mix channel. The mixed output is unchanged.

## [0.2.1] - 2026-08-26

### Added

- `get_feedback_edge()` is now exported from the top-level `dexed` namespace, alongside `get_carriers()`, `get_modulators()` and `get_mod_matrix()`.

### Fixed

- DX7 algorithm 18 (index 17) feedback is now correctly reported as a self-loop on op 2 instead of op 5.
- DX7 algorithm 21 (index 20) feedback is now correctly reported as a self-loop on op 2 instead of op 5.

  In both algorithms the feedback bits sit in the slot for DX7 operator 3 in
  `FmCore::algorithms[32]` (`src/msfa/fm_core.cc`), not operator 6. All 32
  entries in `_ALGORITHM_DATA` now agree with that table on carriers,
  modulators, modulation edges, and feedback edges.

## [0.2.0] - 2026-04-10

### Changed

- **Breaking:** `Algorithm.feedback_op` (int) replaced with `Algorithm.feedback_edge` (Tuple[int, int]) representing `(source, target)`. Most algorithms have `source == target` (self-feedback). Algorithms 3 and 5 (DX7 algos 4 and 6) now correctly represent cross-operator feedback.
- **Breaking:** `get_feedback_op()` replaced with `get_feedback_edge()`.
- **Breaking:** `OperatorGraph.set_feedback(op, level)` signature changed to `set_feedback(source, target, level)`.
- **Breaking:** `OperatorGraph.get_feedback(op)` signature changed to `get_feedback(source, target)`.
- **Breaking:** `OperatorGraph.from_matrix()` feedback parameter changed from `Dict[int, int]` to `Dict[Tuple[int, int], int]`.

### Fixed

- DX7 algorithm 4 (index 3) feedback is now correctly modeled as op 3 -> op 5 instead of a self-loop on op 5.
- DX7 algorithm 6 (index 5) feedback is now correctly modeled as op 4 -> op 5 instead of a self-loop on op 5.
- `OperatorGraph` render methods now correctly route cross-operator feedback: the source operator's output is written to the feedback buffer and the target operator reads from it.

## [0.1.0] - 2026-03-06

Initial release.
