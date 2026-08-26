# dexed-py

Python bindings for the [Dexed](https://github.com/asb2m10/dexed) DX7 synthesizer. Supports high-level patch editing, sysex import/export, and low-level parameter arrays.

[Dexed](https://github.com/asb2m10/dexed) is licensed under the GPL v3. The [msfa](https://github.com/google/music-synthesizer-for-android) component (`src/msfa`) is Apache 2.0 licensed.

## Requirements

- **Python** >= 3.11
- **Platforms**: macOS, Linux, Windows
- **Runtime dependency**: NumPy
- **Optional**: [JAX](https://github.com/jax-ml/jax) — for PyTree integration and `jax.pure_callback` workflows

## Installation

```bash
pip install dexed-py
```

## Quick Start

### Using the Patch API

`Patch` is the human-friendly interface — parameters use native DX7 ranges and string names. Use it for sound design, sysex import/export, and interactive exploration.

```python
from dexed import Patch, DexedSynth

patch = Patch(name="My Sound")
patch.algorithm = 15        # 0-31
patch.feedback = 5
patch.op[0].output_level = 99
patch.op[0].envelope.rates = [99, 85, 35, 50]
patch.op[0].envelope.levels = [99, 75, 0, 0]
patch.lfo.wave = "sine"

synth = DexedSynth(sample_rate=44100)
synth.load_patch(patch)
audio = synth.render(midi_note=60, velocity=100, note_duration=1.0, render_duration=1.5)
```

### Loading DX7 Sysex Files

```python
from dexed import Patch

# Load a 32-voice bank (4096-byte .syx file)
patches = Patch.load_bank("rom1a.syx")
for i, p in enumerate(patches[:5]):
    print(f"  {i}: {p.name.strip()} (algorithm {p.algorithm})")

# Save patches back to a bank file
Patch.save_to_bank("my_bank.syx", patches)
```

### ML / JAX Workflow with Preset

`Preset` is the ML-native representation: a single **(145,) float32** vector covering all synth state. Continuous parameters are normalized to [0, 1]; discrete parameters (algorithm, curves, etc.) are integer-valued. All fields are JAX PyTree data leaves — changing any value, including `algorithm`, **never triggers JIT recompilation**.

```python
import numpy as np
from dexed import Patch, Preset, DexedSynth

# From a sysex bank
preset = Patch.load_bank("rom1a.syx")[0].to_preset()

# Or construct directly (continuous params normalized [0, 1])
preset = Preset(
    algorithm=15,
    feedback=0.5,
    op_output_level=np.full(6, 0.8, dtype=np.float32),
)

synth = DexedSynth()
synth.load_preset(preset)
audio = synth.render(midi_note=60, velocity=100)

# Flat array round-trip
arr     = preset.to_array()  # (145,) float32
preset2 = Preset.from_array(arr)

# Bulk storage: 100k presets ~ 55 MB
bank = np.stack([p.to_array() for p in presets])  # (N, 145)
np.save("bank.npy", bank)
presets = [Preset.from_array(row) for row in np.load("bank.npy")]
```

### JAX pure_callback

The synth can be called non-differentiably from JIT-compiled JAX code via `jax.pure_callback`.

```python
import jax
from jax import numpy as jnp
from dexed import DexedSynth, Preset

SAMPLE_RATE = 44100
NOTE_DURATION = 0.5
RENDER_DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * RENDER_DURATION)

synth = DexedSynth(sample_rate=SAMPLE_RATE)

def render_fn(preset):
    synth.load_preset(preset)
    return synth.render(midi_note=60, velocity=100,
                        note_duration=NOTE_DURATION, render_duration=RENDER_DURATION)

@jax.jit
def jitted_render(preset):
    return jax.pure_callback(
        render_fn, jax.ShapeDtypeStruct((NUM_SAMPLES,), jnp.float32), preset,
    )

audio = jitted_render(Preset(algorithm=0, feedback=0.5))

# Changing algorithm does NOT recompile — all fields are data leaves
audio2 = jitted_render(Preset(algorithm=15, feedback=0.3))
```

For flat-vector policies (Beta distribution over the 145-dim space):

```python
_, treedef = jax.tree.flatten(Preset())  # universal — no meta fields

@jax.jit
def render_from_flat(flat_params):   # (145,) float32
    preset = jax.tree.unflatten(treedef, Preset.array_to_leaves(flat_params))
    return jax.pure_callback(render_fn, jax.ShapeDtypeStruct((NUM_SAMPLES,), jnp.float32), preset)
```

### Algorithm Metadata

```python
from dexed import (
    algorithms, get_carriers, get_modulators, get_mod_matrix, get_feedback_edge,
)

alg = algorithms[15]
print(f"carriers: {alg.carriers}")
print(f"modulators: {alg.modulators}")
print(f"modulation matrix:\n{alg.mod_matrix}")   # 6x6 int8

get_carriers(31)   # [0, 1, 2, 3, 4, 5] — all parallel
```

`feedback_edge` is a `(source, target)` pair of operator indices. Most algorithms
feed an operator back into itself, so `source == target`:

```python
get_feedback_edge(0)    # (5, 5) — DX7 algorithm 1, op 6 into itself
get_feedback_edge(17)   # (2, 2) — DX7 algorithm 18, op 3 into itself
```

Two algorithms are different: their feedback wraps a whole chain rather than a
single operator, so the pair has two distinct operators.

```python
get_feedback_edge(3)    # (3, 5) — DX7 algorithm 4, op 4 back into op 6
get_feedback_edge(5)    # (4, 5) — DX7 algorithm 6, op 5 back into op 6
```

All indices are 0-based, so operator index `i` is DX7 operator `i + 1`.

### Individual Operator Outputs

```python
synth.load_patch(patch)
audio = synth.render_all_ops(midi_note=60)
# audio.shape = (7, T): channels 0-5 are operators 0-5, channel 6 is final mix
```

### Feedback Normalization

By default, `normalize_feedback` is `False`, which preserves Dexed-authentic behavior: algorithms 3, 5, and 31 (DX7 algorithms 4, 6, 32) have reduced feedback strength, matching the original hardware. Set it to `True` for consistent feedback scaling across all 32 algorithms, which is useful when feedback should be comparable regardless of algorithm choice (e.g. in ML pipelines).

```python
synth.normalize_feedback = True  # consistent across all 32 algorithms
```

### Custom Operator Graphs

`OperatorGraph` lets you build arbitrary FM topologies — not limited to the 32 standard DX7 algorithms, and not limited to 6 operators.

```python
from dexed import OperatorGraph

graph = OperatorGraph(num_ops=7)
for i in range(7):
    graph.op[i].output_level = 99
for i in range(6, 0, -1):
    graph.connect(i, i - 1)
graph.set_carriers([0])
graph.set_feedback(6, 6, level=7)

audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                     note_duration=1.0, render_duration=1.5)

# From a modulation matrix
import numpy as np
mod_matrix = np.zeros((4, 4), dtype=np.float32)
mod_matrix[0, 1] = 1.0  # Op 1 modulates Op 0
graph = OperatorGraph.from_matrix(mod_matrix, carriers=[0], feedback={(3, 3): 5})

# From a standard DX7 algorithm (0-indexed)
graph = OperatorGraph.from_algorithm(15)

# Visualize the graph
print(graph.summary())
print(graph.to_ascii())
print(graph.to_mermaid())   # paste into any Mermaid renderer
```

## API Reference

### Patch

```python
patch = Patch(name="My Sound")
patch.algorithm = 15          # 0-31
patch.feedback = 5            # 0-7
patch.osc_key_sync = True
patch.transpose = 24          # 0-48 (24 = C3)

patch.lfo.speed = 35          # 0-99
patch.lfo.delay = 0
patch.lfo.pitch_mod_depth = 0
patch.lfo.amp_mod_depth = 0
patch.lfo.sync = False
patch.lfo.wave = "sine"       # triangle, saw_down, saw_up, square, sine, s&h

patch.pitch_envelope.rates  = [99, 99, 99, 99]
patch.pitch_envelope.levels = [50, 50, 50, 50]

op = patch.op[0]              # 0-indexed: op[0] through op[5]
op.output_level = 99          # 0-99
op.frequency_coarse = 1       # 0-31
op.frequency_fine = 0         # 0-99
op.frequency_mode = 0         # 0=ratio, 1=fixed
op.detune = 7                 # 0-14 (7 = center)
op.velocity_sensitivity = 0   # 0-7
op.amp_mod_sensitivity = 0    # 0-3
op.rate_scaling = 0           # 0-7
op.breakpoint = 39            # 0-99
op.left_depth = 0             # 0-99
op.right_depth = 0            # 0-99
op.left_curve = "lin"         # lin, exp-, exp+, log
op.right_curve = "lin"
op.envelope.rates  = [99, 99, 99, 99]
op.envelope.levels = [99, 99, 99, 0]
```

### Patch — Format Conversion

```python
# Patch <-> Preset
preset = patch.to_preset()
patch  = preset.to_patch()
patch  = Preset.from_patch(patch)  # classmethod alternative

# Sysex (156 bytes unpacked, 128 bytes packed)
sysex  = patch.to_sysex()
patch  = Patch.from_sysex(sysex_bytes)
packed = patch.to_packed()
patch  = Patch.from_packed(packed_bytes)

# Bank (32-voice .syx files)
patches = Patch.load_bank("bank.syx")
Patch.save_to_bank("bank.syx", patches)

# Raw DX7 format (155 integers in native ranges)
raw   = patch.to_raw()
patch = Patch.from_raw(raw_params)
```

### Preset

See [docs/parameter-format.md](docs/parameter-format.md) for the full array layout and the per-operator interface.

```python
from dexed import Preset
import numpy as np

preset = Preset(
    algorithm=15,                                           # 0-31
    feedback=0.71,                                          # 5/7
    osc_key_sync=1,                                         # 0 or 1
    lfo_sync=0,
    lfo_wave=4,                                             # 0-5 (sine)
    op_output_level=np.ones(6, dtype=np.float32),
    op_frequency_mode=np.zeros(6, dtype=np.int32),          # 0=ratio, 1=fixed
    op_left_curve=np.zeros(6, dtype=np.int32),              # 0-3
    op_right_curve=np.zeros(6, dtype=np.int32),
)

arr     = preset.to_array()       # (145,) float32
preset  = Preset.from_array(arr)

# Per-operator decomposition (useful for per-operator ML architectures)
gc = preset.global_continuous()   # (15,)   float32
gi = preset.global_ints()         # (4,)    int32
oc = preset.op_continuous()       # (6, 18) float32
oi = preset.op_ints()             # (6, 3)  int32
preset = Preset.from_operator_bundles(gc, gi, oc, oi)
```

### DexedSynth

```python
synth = DexedSynth(sample_rate=44100)

synth.load_patch(patch)    # from Patch
synth.load_preset(preset)  # from Preset

synth.algorithm            # read-only: currently loaded algorithm (0-31)
synth.normalize_feedback   # bool, read-write (default False)

audio = synth.render(midi_note=60, velocity=100,
                     note_duration=1.0, render_duration=1.5)
audio = synth.render_all_ops(midi_note=60)  # (7, T)
```

### OperatorGraph

```python
graph = OperatorGraph(num_ops=6)

# Connection API (all methods return self for chaining)
graph.connect(source, target, amount=1.0)
graph.disconnect(source, target)
graph.disconnect_all()
graph.set_carriers([0, 2])
graph.set_feedback(source, target, level=7)   # 0 disables, 1-7
                                              # source == target for a self-loop

# Query API
graph.mod_matrix       # NxN float32 (read-only copy)
graph.carriers         # List[int]
graph.modulators       # List[int]
graph.get_connections() # [(source, target, amount), ...]

# Visualization
graph.summary()        # human-readable text
graph.to_ascii()       # ASCII art
graph.to_mermaid()     # Mermaid diagram syntax

# Factory methods
OperatorGraph.from_algorithm(15)
OperatorGraph.from_matrix(mod_matrix, carriers=[0], feedback={5: 7})

# Rendering
audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                     note_duration=1.0, render_duration=1.5)
audio = graph.render_all_ops(midi_note=60)  # (num_ops+1, T)
```

## Building from Source

Requires a C++17 compiler and CMake >= 3.15.

```bash
git clone --recursive https://github.com/DBraun/dexed-py.git
cd dexed-py
pip install -e .
python -m pytest -v tests
```

### Editing the Python sources

The `.py` files are installed by CMake (`install.components = ["python_modules"]`),
so a plain `pip install -e .` copies them into the environment instead of linking
them. Imports keep resolving to that copy — edit `dexed/algorithms.py` and your
tests go on exercising the version from install time. Ask for a rebuild on import
to get a live checkout:

```bash
pip install scikit-build-core nanobind
pip install -e . --no-build-isolation \
  --config-settings=editable.rebuild=true \
  --config-settings=editable.verbose=false \
  --config-settings=build-dir="build/{wheel_tag}"
```

Now both Python and C++ edits take effect on the next import. `build-dir` is
required by rebuild mode and is already covered by `.gitignore`;
`editable.verbose=false` keeps CMake from printing on every import.

To confirm you have a live checkout rather than a copy:

```bash
python -c "import dexed; print(dexed.__version__)"   # edit dexed/version.py, rerun
```
