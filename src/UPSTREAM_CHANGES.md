# Changes from upstream Dexed

This documents modifications to files originally from
[asb2m10/dexed](https://github.com/asb2m10/dexed).

## What is actually compiled

Only these sources are built into the Python extension (see the repo-root
`CMakeLists.txt`, `DEXED_SOURCES`):

```
src/msfa/dx7note.cc   src/msfa/exp2.cc      src/msfa/fm_core.cc
src/msfa/fm_op_kernel.cc  src/msfa/freqlut.cc   src/msfa/lfo.cc
src/msfa/env.cc       src/msfa/pitchenv.cc  src/msfa/sin.cc
src/EngineMkI.cpp     src/EngineOpl.cpp     src/dexed.cpp
```

Those are in sync with upstream `4e21494` (October 15, 2025), which is still the
most recent commit to touch `Source/msfa`. Everything below describes a
deliberate divergence in one of them.

**The rest of `src/` is not compiled.** The vendored JUCE GUI sources
(`PluginProcessor.*`, `PluginEditor.*`, `CartManager.*`, `AlgoDisplay.*`,
`msfa/tuning.cc`, and so on) are kept for reference only and are *not* tracked
against upstream -- several are more than a year behind it. `src/CMakeLists.txt`
is upstream's plugin build file and is likewise unused; the build that matters
is the repo-root `CMakeLists.txt`. Do not read a divergence in those files as
intentional.

## Files that exist only here

- `src/msfa/libMTSClient.h` — a stub standing in for MTS-ESP's real header,
  which lives in upstream's `libs/` and is not vendored. It declares `MTSClient`
  opaque, exactly as the real one does, and stubs `MTS_HasMaster()` (always
  false) and `MTS_NoteToFrequency()`.
- `src/msfa/Tunings.h` — a stub for the Surge tuning library header, likewise
  outside upstream's `Source/`.

Mirroring upstream's `Source/msfa` over `src/msfa` would delete both and break
the build at `dx7note.h` and `tuning.h`.

## `msfa/dx7note.cc`

### Removed: Portamento

Upstream has full portamento/glissando support. We removed it since dexed-py
renders isolated single notes rather than streaming polyphonic MIDI.

Removed code:
- `logfreq_round2semi()` — quantizes pitch to nearest semitone for glissando mode
- `initPortamento()` — copies portamento state between notes
- `porta_curpitch_[]` tracking in `init()` and `compute()`
- Portamento rate calculation in `compute()` using `Porta::rates[]` and
  `ctrls->portamento_enable_cc`

To restore portamento, refer to the upstream `dx7note.cc` and
`porta.cpp`/`porta.h`. The key changes are:
1. Add `porta_curpitch_[op] = freq` in `init()` after computing `basepitch_[op]`
2. Restore the portamento rate calculation block in `compute()` before the
   per-operator freq lookup
3. In `compute()`, use `porta_curpitch_[op]` instead of `basepitch_[op]` for the
   `Freqlut::lookup` call when portamento is active
4. Restore `initPortamento()` and `logfreq_round2semi()`

### Removed: MTS-ESP tuning in `osc_freq()`

Upstream checks for MTS-ESP master tuning in `osc_freq()` via `MTS_HasMaster()`
/ `MTS_NoteToFrequency()`. We simplified `osc_freq()` to always use
`tuning_state_->midinote_to_logfreq()`, since no MTS client is connected in the
standalone binding. The stubs in `src/msfa/libMTSClient.h` keep the remaining
references compiling; the code paths are dead.

`dexed.cpp` passes `nullptr` for `Dx7Note`'s `MTSClient *` parameter at both
construction sites.

### Added: Explicit initialization

- `fb_buf_[0] = 0; fb_buf_[1] = 0;` in the constructor — zero-initializes the
  feedback buffer
- `mpeTimbre = 0; mpePressure = 0;` in `init()` — zero-initializes MPE state
- `mtsFreq = 0;` in `init()` and in `updateBasePitches()` — upstream sets this
  from the MTS branch that was removed above

## `EngineMkI.cpp` / `EngineMkI.h`

### Added: optional per-stage capture in `compute_fb2()` / `compute_fb3()`

Both functions collapse a two- or three-operator feedback chain into a single
sample loop and only ever write out its last stage, so the operators inside the
chain came back as digital silence from `render_all_ops()`. They now take
optional `stage0`/`stage1` pointers that receive each intermediate operator's
output. The pointers default to `nullptr`, so upstream's own call sites are
unchanged and the mixed output is bit-identical.

## Excluded from the build: `msfa/tuning.cc`

`msfa/tuning.cc` is vendored byte-identically to upstream but is **not**
compiled — the repo-root `CMakeLists.txt` leaves it out of `DEXED_SOURCES`
because it depends on JUCE. (Note that `src/CMakeLists.txt`, upstream's own
build file, does list it; that file is unused here.) `msfa/tuning.h` is also
byte-identical to upstream. The concrete `TuningState` the binding renders with
is `struct StdTuning` in `src/dexed.cpp`.

## Whitespace

`msfa/env.cc` and `EngineMkI.cpp` are stored here with LF line endings where
upstream uses CRLF. Diff with `diff -w` when comparing them, or the real changes
are lost in the noise.
