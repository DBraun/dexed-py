# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-08-26

### Added

- `get_feedback_edge()` is exported from the top-level `dexed` namespace,
  alongside `get_carriers()`, `get_modulators()` and `get_mod_matrix()`, and is
  documented in the API reference.

### Changed

- The package now declares its licence correctly. `pyproject.toml` said
  `Apache-2.0`, which applies only to the vendored `src/msfa` component, while
  the project and both bundled `LICENSE` files are GPL. The metadata is now the
  PEP 639 expression `GPL-3.0-or-later`, matching `CITATION.cff`.
- `Patch.save_to_bank()` is a classmethod, matching its sibling
  `Patch.load_bank()` and the form every doc example already used:
  `Patch.save_to_bank("bank.syx", patches)`. It was an instance method, so the
  documented call bound `self` to the filename and raised `AttributeError`
  without writing a file. `patches` is now required; the undocumented
  `patch.save_to_bank(filename)` form, which padded slots 1-31 with empty
  voices, is gone.
- `Patch.save_to_bank()` writes a complete 4104-byte bulk dump -- sysex header,
  4096-byte payload, checksum and `F7` -- instead of the bare 4096-byte payload
  it used to emit under a `.syx` name. Round-tripping a real dump through
  dexed-py no longer strips its MIDI framing.
- `Patch.load_bank()` locates the bulk dump inside the file instead of assuming
  it starts at byte zero, so a bank preceded by another sysex message no longer
  decodes into 32 byte-shifted voices. It verifies the checksum and warns on a
  mismatch, rejects a file with no 32-voice dump in it, and raises `ValueError`
  rather than `IndexError` on an empty file.
- `Patch.name` is no longer stripped when a patch is loaded. It is a 10-byte
  DX7 field and is now always exactly 10 characters, whatever the patch came
  from, so a bank round-trips byte for byte. Call `.strip()` when displaying a
  name, as the documented examples already do.
- `Algorithm` is frozen, `Algorithm.carriers` and `.modulators` are tuples, and
  `Algorithm.mod_matrix` is non-writeable. `get_carriers()` and
  `get_modulators()` return fresh lists, as `get_mod_matrix()` already did.
  They used to hand out the module-level lists themselves, so appending to a
  returned list corrupted the algorithm table for the rest of the process.
  Comparisons like `algorithms[31].carriers == [0, 1, 2, 3, 4, 5]` need to
  compare against a tuple now; `get_carriers(31)` still returns a list.
- `OperatorGraph.render()` and `render_all_ops()` raise on a cyclic modulation
  matrix instead of falling back to bare index order. The fallback decided where
  the one-sample delay landed by operator numbering, and delayed edges that were
  in no cycle at all, so relabelling an equivalent graph changed the audio. Use
  `set_feedback()` to close a loop -- it has explicit delay semantics.
- `OperatorGraph.set_feedback()` validates `level`. A float was stored verbatim
  and then raised `TypeError: unsupported operand type(s) for >>` from inside
  the render loop; `level=99` was silently clamped to 7 and `level=-2` silently
  recorded nothing. Out-of-range values now raise, and a non-integer level
  raises `TypeError` at the call.

### Fixed

- DX7 algorithm 18 (index 17) and algorithm 21 (index 20) feedback are correctly
  reported as a self-loop on op 2 instead of op 5. In both algorithms the
  feedback bits sit in the slot for DX7 operator 3 in `FmCore::algorithms[32]`
  (`src/msfa/fm_core.cc`), not operator 6. All 32 entries in `_ALGORITHM_DATA`
  now agree with that table on carriers, modulators, modulation edges and
  feedback edges, and the test suite checks them against it.
- Pickling a `DexedSynth` keeps the loaded voice. `__getstate__` saved only the
  sample rate, algorithm and `normalize_feedback`, so an unpickled synth
  reported the right algorithm -- itself derived from the voice data it had just
  dropped -- and then raised on the first `render()`. Pickles written by 0.2.0
  and earlier still load.
- Unpacking a packed voice now masks the "don't care" bits and normalizes
  out-of-range bytes, matching Dexed's `Cartridge::unpackProgram`. A corrupt or
  out-of-spec byte used to be handed through verbatim: packed byte 102 = 200
  gave `pitch_envelope.rates[0] = 200` and a `to_preset()` value of 2.02 in a
  field documented as `[0, 1]`, where Dexed yields 72.
- `OperatorGraph.connect(i, i)` and a nonzero diagonal passed to
  `OperatorGraph.from_matrix()` now raise instead of being stored, reported by
  every query method, and then skipped by both render loops. Self-modulation is
  feedback: use `set_feedback(i, i, level)`.
- The raw `_dexed.DexedSynth.load_sysex()` bounds the algorithm and feedback
  bytes, as Dexed's `unpackProgram` does. Byte 134 was copied through unmasked
  and used to index the 32-entry algorithm table, and byte 135 >= 10 gave the
  feedback kernels a negative shift count. Not reachable through the public
  `DexedSynth`, which always passes `Patch.to_sysex()`.
- Loading a patch whose LFO wave field holds 6 or 7 no longer leaves `.wave`
  raising `IndexError`. The 3-bit field encodes only six waves, so the parse
  paths now clamp, as `to_sysex()` already did on the way out.
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
- `Patch.to_sysex()` now clamps out-of-range parameters instead of bit-masking
  them, so a value past the top of its DX7 range saturates rather than wrapping
  to a small one. `algorithm = 32` used to render as algorithm 1 and
  `feedback = 8` as no feedback at all. `Preset.to_patch()` clamps its integer
  fields the same way, matching its documented behaviour.

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
