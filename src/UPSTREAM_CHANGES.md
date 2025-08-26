# Changes from upstream Dexed

This documents modifications to files originally from [asb2m10/dexed](https://github.com/asb2m10/dexed). Last synced with upstream commit `4e21494` (October 15, 2025).

## `msfa/dx7note.cc`

### Removed: Portamento

Upstream has full portamento/glissando support. We removed it since dexed-py renders isolated single notes rather than streaming polyphonic MIDI.

Removed code:
- `logfreq_round2semi()` — quantizes pitch to nearest semitone for glissando mode
- `initPortamento()` — copies portamento state between notes
- `porta_curpitch_[]` tracking in `init()` and `compute()`
- Portamento rate calculation in `compute()` using `Porta::rates[]` and `ctrls->portamento_enable_cc`

To restore portamento, refer to the upstream `dx7note.cc` and `porta.cpp`/`porta.h`. The key changes are:
1. Add `porta_curpitch_[op] = freq` in `init()` after computing `basepitch_[op]`
2. Restore the portamento rate calculation block in `compute()` before the per-operator freq lookup
3. In `compute()`, use `porta_curpitch_[op]` instead of `basepitch_[op]` for the `Freqlut::lookup` call when portamento is active
4. Restore `initPortamento()` and `logfreq_round2semi()`

### Removed: MTS-ESP tuning in `osc_freq()`

Upstream checks for MTS-ESP master tuning in `osc_freq()` via `MTS_HasMaster()` / `MTS_NoteToFrequency()`. We simplified `osc_freq()` to always use `tuning_state_->midinote_to_logfreq()`, since no MTS client is connected in the standalone binding. The MTS stub functions in `tuning.h` ensure this compiles but the code paths were dead.

### Added: Explicit initialization

- `fb_buf_[0] = 0; fb_buf_[1] = 0;` in constructor — zero-initializes feedback buffer
- `mpeTimbre = 0; mpePressure = 0;` in `init()` — zero-initializes MPE state

### Removed: `tuning.cc`

The `tuning.cc` file is excluded from the build (`CMakeLists.txt`) because it depends on JUCE. Tuning stubs are provided in `tuning.h`.
